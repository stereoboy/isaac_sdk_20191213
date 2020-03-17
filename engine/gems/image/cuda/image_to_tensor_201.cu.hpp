/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "engine/gems/cuda_utils/stride_pointer.hpp"

namespace isaac {

// Converts a 3-channel 8-bit image to a to a 32-bit floating point tensor.
// Tensor data is stored channels, rows, columns.
void ImageToTensor201(StridePointer<const unsigned char> rgb, StridePointer<float> result,
                      size_t rows, size_t cols, float factor, float bias);

}  // namespace isaac
