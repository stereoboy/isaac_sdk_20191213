/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/assert.hpp"
#include "engine/core/math/types.hpp"
#include "engine/core/optional.hpp"
#include "engine/core/tensor/element_type.hpp"
#include "engine/core/tensor/tensor.hpp"

namespace isaac {

// A non-template tensor which can hold a const view on a tensor for a fixed storage order, but
// which is not templated on rank or element type.
//
// The Tensor class is a template type which can only be used if the rank and the element type of
// the tensor is known. However some functions can be written without knowning the actual type of
// the tensor. For example a deserialization function does not know yet what kind of tensor will be
// deserialized and the subsequent user might be able work with different types of tensors.
//
// Currently only const view tensors with a fixed storage mode are supported. This covers the
// primary use case of reading tensors from messages.
template <BufferStorageMode Storage>
class UniversalTensorConstView {
 public:
  // Type used for dimension indices
  using index_t = size_t;

  // Type for storing the array of dimensions of the tensor
  using dimensions_t = VectorX<index_t>;

  // Type for storing the buffer data. This is a const view with fixed storage order.
  using buffer_const_view_t = detail::BufferBase<
      detail::TaggedPointer<const byte, std::integral_constant<BufferStorageMode, Storage>>>;

  UniversalTensorConstView() : element_type_(ElementType::kUnknown) {}
  UniversalTensorConstView(ElementType element_type, const dimensions_t dimensions,
                           buffer_const_view_t buffer)
      : element_type_(element_type), dimensions_(dimensions), buffer_(buffer) {}

  template <typename K, index_t Rank>
  UniversalTensorConstView(TensorBase<K, Rank, buffer_const_view_t> view)
      : element_type_(GetElementType<K>()), dimensions_(view.dimensions()), buffer_(view.data()) {}

  // The element type of the tensor
  ElementType element_type() const { return element_type_; }

  // The rank of the tensor. This is also the length of the dimensions.
  index_t rank() const { return dimensions_.size(); }

  // The dimensions of the tensor. Most significant dimension comes first in the array.
  const dimensions_t& dimensions() const { return dimensions_; }

  // The element type of the tensor
  const buffer_const_view_t& buffer() const { return buffer_; }

  // Checks if the tensor has a specific rank and element type
  template <typename K, index_t Rank>
  bool isOfType() {
    return Rank == rank() && GetElementType<K>() == element_type();
  }

  // Tries to get a tensor with specified rank and element type. Returns nullopt if the stored
  // tensor has a different type.
  template <typename K, index_t Rank>
  std::optional<TensorBase<K, Rank, buffer_const_view_t>> tryGet() {
    if (!isOfType<K, Rank>()) {
      return std::nullopt;
    }
    return TensorBase<K, Rank, buffer_const_view_t>(buffer(), dimensions());
  }

  // Gets a tensor with specified rank and element type. Asserts if the stored tensor has a
  // different type.
  template <typename K, index_t Rank>
  TensorBase<K, Rank, buffer_const_view_t> get() {
    const bool ok = isOfType<K, Rank>();
    ASSERT(ok, "Tensor does not have the expected type. Expected: (%s, %d). Actual: (%s, %d).",
           ElementTypeCStr(GetElementType<K>()), Rank,
           ElementTypeCStr(element_type_), dimensions_.size());
    return TensorBase<K, Rank, buffer_const_view_t>(buffer(), dimensions());
  }

  template <typename Tensor>
  bool isOfType() {
    return isOfType<typename Tensor::element_t, Tensor::kRank>();
  }

  template <typename Tensor>
  std::optional<TensorBase<typename Tensor::element_t, Tensor::kRank, buffer_const_view_t>>
  tryGet() {
    return tryGet<typename Tensor::element_t, Tensor::kRank>();
  }

  template <typename Tensor>
  TensorBase<typename Tensor::element_t, Tensor::kRank, buffer_const_view_t> get() {
    return get<typename Tensor::element_t, Tensor::kRank>();
  }

 private:
  ElementType element_type_;
  dimensions_t dimensions_;
  buffer_const_view_t buffer_;
};

// A dynamic tensor const view using host storage
using CpuUniversalTensorConstView = UniversalTensorConstView<BufferStorageMode::Host>;

// A dynamic tensor const view using device storage
using CudaUniversalTensorConstView = UniversalTensorConstView<BufferStorageMode::Cuda>;

}  // namespace isaac
