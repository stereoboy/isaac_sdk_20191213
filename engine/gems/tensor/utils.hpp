/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <algorithm>
#include <array>
#include <initializer_list>
#include <limits>
#include <type_traits>
#include <utility>

#include "engine/core/array/byte_array.hpp"
#include "engine/core/buffers/algorithm.hpp"
#include "engine/core/math/types.hpp"
#include "engine/core/tensor/tensor.hpp"

namespace isaac {

// Copy tensors with compatibale memory layouts
// TODO(dweikersdorf) Find a better implementation to avoid code duplication.
template <typename K, size_t Order, typename SourceContainer, typename TargetContainer>
void Copy(const TensorBase<K, Order,  SourceContainer>& source,
          TensorBase<K, Order, TargetContainer>& target) {
  // Asserts that images have the same shape
  ASSERT(source.dimensions() == target.dimensions(), "Tensor simension mismatch: %zd elements "
         "vs %zd elements", source.element_count(), target.element_count());
  // Copy the bytes
  CopyArrayRaw(source.data().begin(), BufferTraits<SourceContainer>::kStorageMode,
               target.data().begin(), BufferTraits<TargetContainer>::kStorageMode,
               source.data().size());
}
template <typename K, size_t Order, typename SourceContainer>
void Copy(const TensorBase<K, Order,  SourceContainer>& source, TensorView<K, Order> target) {
  // Asserts that images have the same shape
  ASSERT(source.dimensions() == target.dimensions(), "Tensor simension mismatch: %zd elements "
         "vs %zd elements", source.element_count(), target.element_count());
  // Copy the bytes
  CopyArrayRaw(source.data().begin(), BufferTraits<SourceContainer>::kStorageMode,
               target.data().begin(), BufferStorageMode::Host, source.data().size());
}
template <typename K, size_t Order, typename SourceContainer>
void Copy(const TensorBase<K, Order,  SourceContainer>& source, CudaTensorView<K, Order> target) {
  // Asserts that images have the same shape
  ASSERT(source.dimensions() == target.dimensions(), "Tensor simension mismatch: %zd elements "
         "vs %zd elements", source.element_count(), target.element_count());
  // Copy the bytes
  CopyArrayRaw(source.data().begin(), BufferTraits<SourceContainer>::kStorageMode,
               target.data().begin(), BufferStorageMode::Cuda, source.data().size());
}


// Fills a tensor with the given value.
template <typename K, size_t Order,  typename Container>
void Fill(TensorBase<K, Order, Container>& tensor, K value) {
  static_assert(TensorBase<K, Order, Container>::kIsMutable,
                "Cannot Fill const buffer");
  std::fill(tensor.element_wise_begin(), tensor.element_wise_end(), value);
}

// flattens the tensor into a buffer with all stride pitch removed.
template <typename K, size_t Order,  typename Container>
void FlattenData(const TensorBase<K, Order, Container>& tensor, ByteArray* out) {
  ASSERT(out->size() == tensor.data().size(),
         "Buffer must be large enough to hold results");
  std::memcpy(out, tensor.data().begin(), tensor.data().size());
}

// This function takes in a 2 dimensional tensor and crops it using the given range.
// This is a copy operation and the cropped tensor is passed back by reference.
// min_bound is the minimum bound of {row, col} and max_bound is the maximum bound.
// Equivalent to croppedTensor = tensor[min_row: max_row, min_col: max_col] in python for 2
// dimensional tensors.
template <typename K, typename SourceContainer, typename TargetContainer>
void CropTensor2(const TensorBase<K, 2, SourceContainer>& tensor,
                 TensorBase<K, 2, TargetContainer>& croppedTensor,
                 const Vector2i& min_bound, const Vector2i& max_bound) {
  const int min_row = min_bound[0];
  const int min_col = min_bound[1];
  const int max_row = max_bound[0];
  const int max_col = max_bound[1];
  // Check the crop is within range of the tensor.
  const auto tensor_size = tensor.dimensions();
  ASSERT(min_row >= 0 && min_col >= 0 && max_row <= static_cast<int>(tensor_size[0]) &&
             max_col <= static_cast<int>(tensor_size[1]),
         "Crop is out of bounds of the 2d tensor.");

  // Check the size of  the crop is positive.
  const Vector2i target_size = max_bound - min_bound;
  ASSERT(target_size[0] > 0 && target_size[1] > 0, "Crop size smaller than 0.");

  // Crop the tensor.
  croppedTensor.resize(target_size[0], target_size[1]);
  for (int row = min_row; row < max_row; row++) {
    for (int col = min_col; col < max_col; col++) {
      croppedTensor(row - min_row, col - min_col) = tensor(row, col);
    }
  }
}
}  // namespace isaac
