/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <array>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

#include "engine/core/assert.hpp"
#include "engine/core/buffers/buffer.hpp"
#include "engine/core/byte.hpp"
#include "engine/core/math/types.hpp"
#include "engine/gems/math/float16.hpp"

namespace isaac {

// Tensor of type K with order Order and specified storage type
// This class is primarily intended to be used as a container
// and does not support math operations.
// Free functions to handle math or other tensor operations may exist as utilities.
//
// K is the data type of the tensor.
// Order is fixed at compile time, but dimensions are dynamic
// buffer_t represents the storage type of the data.
//
template <typename K, size_t Order, typename BufferType>
class TensorBase {
  static_assert(Order > 0, "Isaac does not support order 0 tensors");

 public:
  using buffer_t = BufferType;

  static constexpr bool kIsMutable = BufferTraits<buffer_t>::kIsMutable;
  static constexpr bool kIsOwning = BufferTraits<buffer_t>::kIsOwning;

  static constexpr int kRank = Order;

  using element_t = std::remove_cv_t<K>;
  using element_const_ptr_t = std::add_const_t<element_t>*;
  using element_ptr_t = std::conditional_t<kIsMutable, element_t*, element_const_ptr_t>;

  using element_const_ref_t = std::add_const_t<element_t>&;
  using element_ref_t = std::conditional_t<kIsMutable, element_t&, element_const_ref_t>;

  using raw_const_ptr_t = std::add_const_t<byte>*;
  using raw_ptr_t = std::conditional_t<kIsMutable, byte*, raw_const_ptr_t>;

  using buffer_view_t = typename BufferTraits<buffer_t>::buffer_view_t;
  using buffer_const_view_t = typename BufferTraits<buffer_t>::buffer_const_view_t;

  using tensor_view_t = TensorBase<K, Order, buffer_view_t>;
  using tensor_const_view_t = TensorBase<K, Order, buffer_const_view_t>;

  using dimension_t = size_t;
  using dimension_array_t = Vector<dimension_t, Order>;

  using index_t = size_t;
  using index_array_t = Vector<index_t, Order>;

  TensorBase() {
    // If no dimensions are provided set to zero.
    setDimensions(dimension_array_t::Constant(0));
  }

  // Constructs a tensor with the given dimensions. Allocates storage.
  TensorBase(const dimension_array_t& dimensions) { resize(dimensions); }
  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 1, bool> = true>
  TensorBase(dimension_t dim1) : TensorBase(dimension_array_t{dim1}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 2, bool> = true>
  TensorBase(dimension_t dim1, dimension_t dim2) : TensorBase(dimension_array_t{dim1, dim2}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 3, bool> = true>
  TensorBase(dimension_t dim1, dimension_t dim2, dimension_t dim3)
      : TensorBase(dimension_array_t{dim1, dim2, dim3}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 4, bool> = true>
  TensorBase(dimension_t dim1, dimension_t dim2, dimension_t dim3, dimension_t dim4)
      : TensorBase(dimension_array_t{dim1, dim2, dim3, dim4}) {}

  // Constructor of tensor around existing data. Does not allocate storage.
  TensorBase(buffer_t data, const dimension_array_t& dimensions)
      : data_(std::move(data)) {
    ASSERT(data.size() >= this->data().size(), "Provided buffer is too small");
    ASSERT(CheckDimensions(dimensions), "Trying to create tensor with invalid dimensions");
    setDimensions(dimensions);
  }

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 1, bool> = true>
  TensorBase(buffer_t data, dimension_t dim1)
      : TensorBase(std::move(data), dimension_array_t{dim1}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 2, bool> = true>
  TensorBase(buffer_t data, dimension_t dim1, dimension_t dim2)
      : TensorBase(std::move(data), dimension_array_t{dim1, dim2}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 3, bool> = true>
  TensorBase(buffer_t data, dimension_t dim1, dimension_t dim2, dimension_t dim3)
      : TensorBase(std::move(data), dimension_array_t{dim1, dim2, dim3}) {}

  // Helper constructor of the appropriate order.
  template <size_t Dummy = Order, typename std::enable_if_t<Dummy == 4, bool> = true>
  TensorBase(buffer_t data, dimension_t dim1, dimension_t dim2, dimension_t dim3, dimension_t dim4)
      : TensorBase(std::move(data), dimension_array_t{dim1, dim2, dim3, dim4}) {}

  // Copy construction uses the default behavior
  TensorBase(const TensorBase& other) = default;
  // Copy assignment uses the default behavior1
  TensorBase& operator=(const TensorBase& other) = default;
  // Move construction uses the default behavior
  TensorBase(TensorBase&& other) = default;
  // Move assignment uses the default behavior
  TensorBase& operator=(TensorBase&& other) = default;

  // Create a view if the data is mutable
  template <bool X = kIsMutable>
  std::enable_if_t<X, tensor_view_t> view() {
    return tensor_view_t({this->data().begin(), this->data().size()},
                         this->dimensions());
  }

  // create a const view
  tensor_const_view_t const_view() const {
    return tensor_const_view_t({this->data().begin(), this->data().size()},
                               this->dimensions());
  }

  // Allow conversion from owning to mutable view
  template <bool X = kIsMutable>
  operator std::enable_if_t<X, tensor_view_t>() {
    return view();
  }
  // Allow conversion to const view
  operator tensor_const_view_t() const { return const_view(); }

  // Helper function to compute the position of an element in the tensor
  size_t indexToOffset(const index_array_t& indices) const {
    return offsets_.dot(indices);
  }

  // Const Pointer to the beginning of the data block
  element_const_ptr_t element_wise_begin() const {
    return reinterpret_cast<element_const_ptr_t>(data_.begin());
  }
  // Pointer to the beginning of the data block
  element_ptr_t element_wise_begin() {
    return reinterpret_cast<element_ptr_t>(data_.begin());
  }
  // Const Pointer to the beginning of the data block
  element_const_ptr_t element_wise_end() const {
    return reinterpret_cast<element_const_ptr_t>(data_.end());
  }
  // Pointer to the beginning of the data block
  element_ptr_t element_wise_end() {
    return reinterpret_cast<element_ptr_t>(data_.end());
  }

  // const access to the underlying buffer object
  const buffer_t& data() const { return data_; }
  // access to the underlying buffer object
  template <bool X = kIsMutable>
  std::enable_if_t<X, buffer_t&> data() {
    return data_;
  }

  // The total number of elements in the tensor
  size_t element_count() const {
    return offsets_[0] * dimensions_[0];
  }

  // The total numbe of bytes required to store the tensor
  size_t byte_size() const {
    return element_count() * sizeof(element_t);
  }

  // Returns the order of the tensor
  constexpr size_t order() const { return Order; }
  // Returns the dimensions of the tensor
  dimension_array_t dimensions() const { return dimensions_; }

  // Resizes the tensor memory, This operation is destructive.
  template <bool X = kIsOwning>
  std::enable_if_t<X, void> resize(const dimension_array_t& dimensions) {
    ASSERT(CheckDimensions(dimensions), "Invalid dimensions");
    setDimensions(dimensions);
    data_ = buffer_t(byte_size());
  }

  // Helper function of the approrpriate order for syntatic nicety.
  template <bool X = (Order == 1)>
  std::enable_if_t<X, void> resize(dimension_t dimension1) {
    return resize(dimension_array_t{dimension1});
  }
  // Helper function of the approrpriate order for syntatic nicety.
  template <bool X = (Order == 2)>
  std::enable_if_t<X, void> resize(dimension_t dimension1, dimension_t dimension2) {
    return resize(dimension_array_t{dimension1, dimension2});
  }
  // Helper function of the approrpriate order for syntatic nicety.
  template <bool X = (Order == 3)>
  std::enable_if_t<X, void> resize(dimension_t dimension1, dimension_t dimension2,
                                   dimension_t dimension3) {
    return resize(dimension_array_t{dimension1, dimension2, dimension3});
  }
  // Helper function of the approrpriate order for syntatic nicety.
  template <bool X = (Order == 4)>
  std::enable_if_t<X, void> resize(dimension_t dimension1, dimension_t dimension2,
                                   dimension_t dimension3, size_t dimension4) {
    return resize(dimension_array_t{dimension1, dimension2, dimension3, dimension4});
  }

  // Accesses the given index in the tensor with respect to memory ordering
  element_t operator()(const index_array_t& indicies) const {
    return *(this->element_wise_begin() + indexToOffset(indicies));
  }

  // Accesses the given index in the tensor with respect to memory ordering
  template <bool X = kIsMutable>
  std::enable_if_t<X, element_ref_t> operator()(const index_array_t& indicies) {
    return *(this->element_wise_begin() + indexToOffset(indicies));
  }

  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 1)>
  std::enable_if_t<X, element_t> operator()(index_t index1) const {
    return operator()(index_array_t{index1});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 1), bool Y = kIsMutable>
  std::enable_if_t<X && Y, element_ref_t> operator()(index_t index1) {
    return operator()(index_array_t{index1});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 2)>
  std::enable_if_t<X, element_t> operator()(index_t index1, index_t index2) const {
    return operator()(index_array_t{index1, index2});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 2), bool Y = kIsMutable>
  std::enable_if_t<X && Y, element_ref_t> operator()(index_t index1, index_t index2) {
    return operator()(index_array_t{index1, index2});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 3)>
  std::enable_if_t<X, element_t> operator()(index_t index1, index_t index2, index_t index3) const {
    return operator()(index_array_t{index1, index2, index3});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 3), bool Y = kIsMutable>
  std::enable_if_t<X && Y, element_ref_t> operator()(index_t index1, index_t index2,
                                                     index_t index3) {
    return operator()(index_array_t{index1, index2, index3});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 4)>
  std::enable_if_t<X, element_t> operator()(index_t index1, index_t index2, index_t index3,
                                            index_t index4) const {
    return operator()(index_array_t{index1, index2, index3, index4});
  }
  // Helper wrappers for operator access for the most common orders of tensors
  template <bool X = (Order == 4), bool Y = kIsMutable>
  std::enable_if_t<X && Y, element_ref_t> operator()(index_t index1, index_t index2, index_t index3,
                                                     index_t index4) {
    return operator()(index_array_t{index1, index2, index3, index4});
  }

  // Creates a view on a slice with one dimension less. The dimension with highest significance
  // is sliced.
  template <bool X = kIsMutable && (Order >= 2)>
  std::enable_if_t<X, TensorBase<K, Order - 1, buffer_view_t>> slice(index_t index) {
    using slice_t = TensorBase<K, Order - 1, buffer_view_t>;
    const size_t slice_size = static_cast<size_t>(offsets_[0]) * sizeof(K);
    const size_t slice_offset = static_cast<size_t>(index) * slice_size;
    return slice_t(buffer_view_t{this->data().begin() + slice_offset, slice_size},
                   this->dimensions().template segment<Order - 1>(1).eval());
  }
  template <bool X = !kIsMutable && (Order >= 2)>
  std::enable_if_t<X, TensorBase<K, Order - 1, buffer_const_view_t>> slice(index_t index) {
    return const_slice(index);
  }

  // Creates a const view on a slice with one dimension less. The dimension with highest
  // significance is sliced.
  template <bool X = (Order >= 2)>
  TensorBase<K, Order - 1, buffer_const_view_t> const_slice(index_t index) const {
    using const_slice_t = TensorBase<K, Order - 1, buffer_const_view_t>;
    const size_t slice_size = static_cast<size_t>(offsets_[0]) * sizeof(K);
    const size_t slice_offset = static_cast<size_t>(index) * slice_size;
    return const_slice_t(buffer_const_view_t{this->data().begin() + slice_offset, slice_size},
                         this->dimensions().template segment<Order - 1>(1).eval());
  }

 private:
  // Sets the dimensions of the tensor and updates offsets.
  void setDimensions(const dimension_array_t& dimensions) {
    dimensions_ = dimensions;
    offsets_[Order - 1] = 1;  // Order is guaranteed to be positive
    for (size_t i = Order - 1; i > 0; i--) {
      offsets_[i - 1] = offsets_[i] * dimensions_[i];
    }
  }

  // Make sure the dimensions are greater or equal than 0
  static bool CheckDimensions(const dimension_array_t& dimensions) {
    return (dimensions.array() >= 0).all();
  }

  // storage for the tensor
  buffer_t data_;
  // The dimensions of the tensor.
  dimension_array_t dimensions_;
  // Dimension offsets for indexing rows of storage
  dimension_array_t offsets_;
};  // namespace isaac

template <typename K, size_t Order>
using Tensor = TensorBase<K, Order, CpuBuffer>;

template <typename K, size_t Order>
using TensorView = TensorBase<K, Order, CpuBufferView>;

template <typename K, size_t Order >
using TensorConstView = TensorBase<K, Order, CpuBufferConstView>;

#define ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, T, S) \
  using Tensor##N##S = Tensor<T, N>;             \
  using TensorView##N##S = TensorView<T, N>;     \
  using TensorConstView##N##S = TensorConstView<T, N>;

#define ISAAC_DECLARE_TENSOR_TYPES(N)                \
  template <class K>                                 \
  using Tensor##N = Tensor<K, N>;                    \
  template <class K>                                 \
  using TensorView##N = TensorView<K, N>;            \
  template <class K>                                 \
  using TensorConstView##N = TensorConstView<K, N>;  \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, uint8_t, ub)    \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, uint16_t, ui16) \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, int, i)         \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, double, d)      \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, float, f)       \
  ISAAC_DECLARE_TENSOR_TYPES_IMPL(N, float16, f16)

ISAAC_DECLARE_TENSOR_TYPES(1)
ISAAC_DECLARE_TENSOR_TYPES(2)
ISAAC_DECLARE_TENSOR_TYPES(3)
ISAAC_DECLARE_TENSOR_TYPES(4)

// -------------------------------------------------------------------------------------------------

// An Tensor stored in device memory which owns it's memory
template <class K, size_t Order>
using CudaTensor = TensorBase<K, Order, CudaBuffer>;

// A mutable view on an Tensor which is stored on GPU device memory, does not own memory, but can
// be used to read and write the data of the underlying Tensor.
template <class K, size_t Order>
using CudaTensorView = TensorBase<K, Order, CudaBufferView>;

// A non-mutable view on an Tensor which is stored on GPU device memory, does not own its memory,
// and can only be used to read the data of the underlying Tensor.
template <class K, size_t Order>
using CudaTensorConstView = TensorBase<K, Order, CudaBufferConstView>;

// Helper macro for ISAAC_DECLARE_CUDA_TENSOR_TYPES
#define ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, K, S) \
  using CudaTensor##N##S = CudaTensor<K, N>;          \
  using CudaTensorView##N##S = CudaTensorView<K, N>;  \
  using CudaTensorConstView##N##S = CudaTensorConstView<K, N>;

// Helper macro to define various CudaTensor types
#define ISAAC_DECLARE_CUDA_TENSOR_TYPES(N)                  \
  template <class K>                                        \
  using CudaTensor##N = CudaTensor<K, N>;                   \
  template <class K>                                        \
  using CudaTensorView##N = CudaTensorView<K, N>;           \
  template <class K>                                        \
  using CudaTensorConstView##N = CudaTensorConstView<K, N>; \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, uint8_t, ub)      \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, uint16_t, ui16)   \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, int, i)           \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, double, d)        \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, float, f)         \
  ISAAC_DECLARE_CUDA_TENSOR_TYPES_IMPL(N, float16, f16)

ISAAC_DECLARE_CUDA_TENSOR_TYPES(1)
ISAAC_DECLARE_CUDA_TENSOR_TYPES(2)
ISAAC_DECLARE_CUDA_TENSOR_TYPES(3)
ISAAC_DECLARE_CUDA_TENSOR_TYPES(4)

}  // namespace isaac
