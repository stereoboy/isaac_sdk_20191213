/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "conversions.hpp"

#include <algorithm>

#include "engine/core/math/utils.hpp"
#include "engine/gems/image/color.hpp"
#include "engine/gems/image/cuda/image_to_tensor_012.cu.hpp"
#include "engine/gems/image/cuda/image_to_tensor_201.cu.hpp"

namespace isaac {

namespace {

// Use a bitshift to quickly divide a number by 256.
uint8_t Div256(int32_t val) {
  return static_cast<uint8_t>(val >> 8);
}

// R = 1.164*(Y' - 16) + 1.596*(Cr - 128)
// G = 1.164*(Y' - 16) - 0.813*(Cr - 128) - 0.391*(Cb - 128)
// B = 1.164*(Y' - 16)                    + 2.018*(Cb - 128)
//
// The coefficients below are the numbers from YCbCr-to-RGB conversion matrix
// converted to 8:8 fixed point numbers.
//
// See: https://en.wikipedia.org/wiki/YCbCr, https://en.wikipedia.org/wiki/YUV
//
void YCbCrToRgb(uint8_t y, uint8_t cb, uint8_t cr, uint8_t* out) {
  const int32_t y_shifted = y - 16;
  const int32_t cr_shifted = cr - 128;
  const int32_t cb_shifted = cb - 128;

  out[0] = Div256(Clamp(298 * y_shifted + 409 * cr_shifted, 0, 0xffff));
  out[1] = Div256(Clamp(298 * y_shifted - 208 * cr_shifted - 100 * cb_shifted, 0, 0xffff));
  out[2] = Div256(Clamp(298 * y_shifted + 516 * cb_shifted, 0, 0xffff));
}

// Converts one image into another image
template <typename K1, int N1, typename Container1, typename K2, int N2, typename Container2,
          typename F>
void ConvertImpl(const ImageBase<K1, N1, Container1>& source, ImageBase<K2, N2, Container2>& target,
                 F op) {
  ASSERT(source.rows() == target.rows(), "row count mismatch: %zd vs. %zd", source.rows(),
         target.rows());
  ASSERT(source.cols() == target.cols(), "col count mismatch: %zd vs. %zd", source.cols(),
         target.cols());
  for (size_t row = 0; row < source.rows(); row++) {
    const K1* it_source = source.row_pointer(row);
    const K1* it_source_end = it_source + N1 * source.cols();
    K2* it_target = target.row_pointer(row);
    for (; it_source != it_source_end; it_source += N1, it_target += N2) {
      op(it_source, it_target);
    }
  }
}

}  // namespace

void ConvertYuyvToRgb(const Image2ub& yuyv, Image3ub& rgb) {
  ASSERT(yuyv.num_pixels() % 2 == 0, "invalid input image size");
  rgb.resize(yuyv.rows(), yuyv.cols());

  // Each 4 byte block encodes two pixels, so we read 4 bytes at a time.
  const uint32_t* raw_in = reinterpret_cast<const uint32_t*>(yuyv.element_wise_begin());
  const uint32_t* raw_end = reinterpret_cast<const uint32_t*>(yuyv.element_wise_end());
  uint8_t* raw_out = rgb.element_wise_begin();

  // Convert each 2 pixel block.
  for (; raw_in != raw_end; ++raw_in) {
    const uint32_t yuvu_block = *raw_in;
    const uint8_t y0 = yuvu_block & 0xff;
    const uint8_t cb = (yuvu_block >> 8) & 0xff;
    const uint8_t y1 = (yuvu_block >> 16) & 0xff;
    const uint8_t cr = (yuvu_block >> 24) & 0xff;

    // Encode first pixel.
    YCbCrToRgb(y0, cb, cr, raw_out);
    raw_out += 3;

    // Encode second pixel.
    YCbCrToRgb(y1, cb, cr, raw_out);
    raw_out += 3;
  }
}

Pixel3f RgbToHsv(const Pixel3f& rgb) {
  const float max = std::max({rgb[0], rgb[1], rgb[2]});
  const float min = std::min({rgb[0], rgb[1], rgb[2]});
  const float delta = max - min;
  Pixel3f hsv{0.0f, 0.0f, max * 255.0f};
  if (delta == 0.0f) {
    return hsv;
  }
  hsv[1] = (delta / max) * 255.0f;

  // hue
  if (max == rgb[0]) {
    hsv[0] = 60.0f * ((rgb[1] - rgb[2]) / delta);
  } else if (max == rgb[1]) {
    hsv[0] = 60.0f * ((rgb[2] - rgb[0]) / delta) + 120.0f;
  } else {
    hsv[0] = 60.0f * ((rgb[0] - rgb[1]) / delta) + 240.0f;
  }

  if (hsv[0] < 0.0f) hsv[0] += 360.0f;
  return hsv;
}

void ConvertRgbaToRgb(ImageConstView4f source, ImageView3ub target) {
  ConvertImpl(source, target, [](const float* src, uint8_t* dst) {
    dst[0] = static_cast<uint8_t>(src[0] * 255.0f + 0.5f);
    dst[1] = static_cast<uint8_t>(src[1] * 255.0f + 0.5f);
    dst[2] = static_cast<uint8_t>(src[2] * 255.0f + 0.5f);
  });
}

void ConvertRgbaToRgb(ImageConstView4ub source, ImageView3ub target) {
  ConvertImpl(source, target, [](const uint8_t* src, uint8_t* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
  });
}

void ConvertBgraToRgb(ImageConstView4ub source, ImageView3ub target) {
  ConvertImpl(source, target, [](const uint8_t* src, uint8_t* dst) {
    dst[0] = src[2];
    dst[1] = src[1];
    dst[2] = src[0];
  });
}

void ConvertRgbToRgba(ImageConstView3ub source, ImageView4ub target, uint8_t alpha) {
  ConvertImpl(source, target, [alpha](const uint8_t* src, uint8_t* dst) {
    dst[0] = src[0];
    dst[1] = src[1];
    dst[2] = src[2];
    dst[3] = alpha;
  });
}

void ConvertUi16ToF32(ImageConstView1ui16 source, ImageView1f target, float scale) {
  ConvertImpl(source, target, [scale](const uint16_t* src, float* dst) {
    *dst = static_cast<float>(*src) * scale;
  });
}

void ConvertF32ToUi16(ImageConstView1f source, ImageView1ui16 target, float scale) {
  const float scale_inv = 1.0f / scale;
  ConvertImpl(source, target, [=](const float* src, uint16_t* dst) {
    *dst = static_cast<uint16_t>(Clamp(*src * scale_inv, 0.0f, 65535.0f));
  });
}

void ImageToNormalizedTensor(ImageConstView3ub rgb_image, Tensor3f& tensor,
                             ImageToTensorIndexOrder index_order,
                             ImageToTensorNormalization normalization) {
  const size_t rows = rgb_image.rows();
  const size_t cols = rgb_image.cols();
  if (index_order == ImageToTensorIndexOrder::k012) {
    tensor.resize(rows, cols, 3);
  } else if (index_order == ImageToTensorIndexOrder::k201) {
    tensor.resize(3, rows, cols);
  } else {
    PANIC("Invalid index order mode: %d", index_order);
  }

  for (size_t row = 0; row < rows; row++) {
    for (size_t col = 0; col < cols; col++) {
      PixelConstRef3ub rgb = rgb_image(row, col);
      float cr, cg, cb;
      if (normalization == ImageToTensorNormalization::kPositiveNegative) {
        cr = (static_cast<float>(rgb[0]) * (2.0f / 255.0f)) - 1.0f;
        cg = (static_cast<float>(rgb[1]) * (2.0f / 255.0f)) - 1.0f;
        cb = (static_cast<float>(rgb[2]) * (2.0f / 255.0f)) - 1.0f;
      } else if (normalization == ImageToTensorNormalization::kUnit) {
        cr = (static_cast<float>(rgb[0]) * (1.0f / 255.0f));
        cg = (static_cast<float>(rgb[1]) * (1.0f / 255.0f));
        cb = (static_cast<float>(rgb[2]) * (1.0f / 255.0f));
      } else if (normalization == ImageToTensorNormalization::kNone) {
        cr = static_cast<float>(rgb[0]);
        cg = static_cast<float>(rgb[1]);
        cb = static_cast<float>(rgb[2]);
      } else if (normalization == ImageToTensorNormalization::kHalfAndHalf) {
        cr = (static_cast<float>(rgb[0]) * (1.0f / 255.0f)) - 0.5f;
        cg = (static_cast<float>(rgb[1]) * (1.0f / 255.0f)) - 0.5f;
        cb = (static_cast<float>(rgb[2]) * (1.0f / 255.0f)) - 0.5f;
      } else {
        PANIC("Invalid image normalization mode: %d", normalization);
      }
      if (index_order == ImageToTensorIndexOrder::k012) {
        tensor(row, col, 0) = cr;
        tensor(row, col, 1) = cg;
        tensor(row, col, 2) = cb;
      } else if (index_order == ImageToTensorIndexOrder::k201) {
        tensor(0, row, col) = cr;
        tensor(1, row, col) = cg;
        tensor(2, row, col) = cb;
      }
    }
  }
}

void ImageToNormalizedTensor(CudaImageConstView3ub rgb_image, CudaTensorView3f result,
                             ImageToTensorIndexOrder index_order,
                             ImageToTensorNormalization normalization) {
  float factor, bias;
  switch (normalization) {
    case ImageToTensorNormalization::kPositiveNegative: factor = 2.0f / 255.0f; bias = -1.0f; break;
    case ImageToTensorNormalization::kUnit:             factor = 1.0f / 255.0f; bias =  0.0f; break;
    case ImageToTensorNormalization::kNone:             factor = 1.0f;          bias =  0.0f; break;
    case ImageToTensorNormalization::kHalfAndHalf:      factor = 1.0f / 255.0f; bias = -0.5f; break;
    default: PANIC("Invalid image normalization mode: %d", normalization); break;
  }

  if (index_order == ImageToTensorIndexOrder::k012) {
    ASSERT(rgb_image.rows() == result.dimensions()[0], "row count mismatch");
    ASSERT(rgb_image.cols() == result.dimensions()[1], "col count mismatch");
    ASSERT(3 == result.dimensions()[2], "channel count mismatch");
    ImageToTensor012({rgb_image.element_wise_begin(), rgb_image.getStride()},
                     {result.element_wise_begin(), rgb_image.cols() * 3 * sizeof(float)},
                     rgb_image.rows(), rgb_image.cols(), factor, bias);
  } else if (index_order == ImageToTensorIndexOrder::k201) {
    ASSERT(rgb_image.rows() == result.dimensions()[1], "row count mismatch");
    ASSERT(rgb_image.cols() == result.dimensions()[2], "col count mismatch");
    ASSERT(3 == result.dimensions()[0], "channel count mismatch");
    ImageToTensor201({rgb_image.element_wise_begin(), rgb_image.getStride()},
                     {result.element_wise_begin(), rgb_image.cols() * sizeof(float)},
                     rgb_image.rows(), rgb_image.cols(), factor, bias);
  } else {
    PANIC("Invalid index order mode: %d", index_order);
  }
}

void NormalizedTensorToImage(TensorConstView3f tensor, ImageToTensorNormalization normalization,
                             Image3ub& rgb_image) {
  const size_t tensor_rows = tensor.dimensions()[0];
  const size_t tensor_cols = tensor.dimensions()[1];
  rgb_image.resize(tensor_rows, tensor_cols);
  if (normalization == ImageToTensorNormalization::kPositiveNegative) {
    for (size_t row = 0; row < tensor_rows; row++) {
      for (size_t col = 0; col < tensor_cols; col++) {
        for (size_t channel = 0; channel < 3; channel++) {
          rgb_image(row, col)[channel] =
              static_cast<uint8_t>((tensor(row, col, channel) / 2.0f + 0.5f) * 255.0f);
        }
      }
    }
  } else if (normalization == ImageToTensorNormalization::kUnit) {
    for (size_t row = 0; row < tensor_rows; row++) {
      for (size_t col = 0; col < tensor_cols; col++) {
        for (size_t channel = 0; channel < 3; channel++) {
          rgb_image(row, col)[channel] = static_cast<uint8_t>(tensor(row, col, channel) * 255.0f);
        }
      }
    }
  } else if (normalization == ImageToTensorNormalization::kNone) {
    for (size_t row = 0; row < tensor_rows; row++) {
      for (size_t col = 0; col < tensor_cols; col++) {
        for (size_t channel = 0; channel < 3; channel++) {
          rgb_image(row, col)[channel] = static_cast<uint8_t>(tensor(row, col, channel));
        }
      }
    }
  } else if (normalization == ImageToTensorNormalization::kHalfAndHalf) {
    for (size_t row = 0; row < tensor_rows; row++) {
      for (size_t col = 0; col < tensor_cols; col++) {
        for (size_t channel = 0; channel < 3; channel++) {
          rgb_image(row, col)[channel] =
              static_cast<uint8_t>((tensor(row, col, channel) + 0.5f) * 255.0f);
        }
      }
    }
  } else {
    PANIC("Invalid noormalization method: %d", normalization);
  }
}

}  // namespace isaac
