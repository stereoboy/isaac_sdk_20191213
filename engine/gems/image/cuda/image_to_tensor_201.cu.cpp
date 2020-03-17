/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "image_to_tensor_201.cu.hpp"

#include <cuda_runtime.h>

namespace isaac {

namespace {
// Cuda implementation of the memory copy. Converts a rgb linear array into a planar
// r, g, b array. Normalizes data to be in the range [0..1]
__global__ void ImageToTensor201Impl(StridePointer<const unsigned char> rgb,
                                     StridePointer<float> result, size_t rows, size_t cols,
                                     float factor, float bias) {
  const int row = blockIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols) return;

  while (col < cols) {
    const unsigned char* data = rgb.row_pointer(row) + 3 * col;
    // Data is laid out in planar blocks by row.
    float* result_data0 = result.row_pointer(row) + col;
    float* result_data1 = result.row_pointer(row + rows) + col;
    float* result_data2 = result.row_pointer(row + 2 * rows) + col;
    // Normalize and write results
    *result_data0 = (float)(data[0]) * factor + bias;
    *result_data1 = (float)(data[1]) * factor + bias;
    *result_data2 = (float)(data[2]) * factor + bias;
    col += blockDim.x * gridDim.x;
  }
}

}  // namespace

void ImageToTensor201(StridePointer<const unsigned char> rgb, StridePointer<float> result,
                      size_t rows, size_t cols, float factor, float bias) {
  dim3 block(256, 1, 1);
  dim3 grid(1, rows, 1);
  ImageToTensor201Impl<<<grid, block>>>(rgb, result, rows, cols, factor, bias);
}

}  // namespace isaac
