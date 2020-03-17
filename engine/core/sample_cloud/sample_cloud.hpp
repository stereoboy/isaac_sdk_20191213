/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <Eigen/Eigen>

#include "engine/core/math/types.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "engine/gems/math/float16.hpp"

namespace isaac {

// Represents a storage container for collection of data samples.  Sample clouds are
// not presumed to have spatial representation by default. They only represent
// some arbitrary data defined on a per sample basis.
//
// For instance a sample cloud could be a list of colors that can be applied to
// another sample cloud representing samples in space or velocity vectors.
//
// Sample clouds have a fixed number of data channels, but have a dynamic number of
// samples specified at runtime.
//
// Channels will be interleaved in memory
template <typename K, size_t Channels, typename BufferType>
class SampleCloudBase {
 public:
  using buffer_t = BufferType;
  static constexpr bool kIsMutable = BufferTraits<buffer_t>::kIsMutable;
  static constexpr bool kIsOwning = BufferTraits<buffer_t>::kIsOwning;

  using data_t = TensorBase<K, 2, buffer_t>;

  using element_t = typename data_t::element_t;
  using element_ptr_t = typename data_t::element_ptr_t;
  using element_const_ptr_t = typename data_t::element_const_ptr_t;

  using buffer_view_t = typename BufferTraits<buffer_t>::buffer_view_t;
  using buffer_const_view_t = typename BufferTraits<buffer_t>::buffer_const_view_t;

  using sample_cloud_view_t = SampleCloudBase<K, Channels, buffer_view_t>;
  using sample_cloud_const_view_t = SampleCloudBase<K, Channels, buffer_const_view_t>;

  using index_t = size_t;

  using eigen_sample_t = Vector<K, Channels>;
  using eigen_const_sample_t = const Vector<K, Channels>;
  using eigen_sample_view_t = Eigen::Map<eigen_sample_t>;
  using eigen_sample_const_view_t = Eigen::Map<eigen_const_sample_t>;
  using eigen_matrix_view_t = Eigen::Map<Matrix<K, Channels, Eigen::Dynamic>>;
  using eigen_matrix_const_view_t = Eigen::Map<const Matrix<K, Channels, Eigen::Dynamic>>;

  SampleCloudBase() {}
  SampleCloudBase(size_t number_samples) : sample_count_(number_samples) { resize(number_samples); }
  SampleCloudBase(buffer_t data, size_t number_samples) : sample_count_(number_samples) {
    data_ = data_t(data, number_samples, Channels);
  }

  // Copy construction uses the default behavior
  SampleCloudBase(const SampleCloudBase& other) = default;
  // Copy assignment uses the default behavior1
  SampleCloudBase& operator=(const SampleCloudBase& other) = default;
  // Move construction uses the default behavior
  SampleCloudBase(SampleCloudBase&& other) = default;
  // Move assignment uses the default behavior
  SampleCloudBase& operator=(SampleCloudBase&& other) = default;

  // Create a view if the data is mutable
  template <bool X = kIsMutable>
  std::enable_if_t<X, sample_cloud_view_t> view() {
    return sample_cloud_view_t({this->data_.data().pointer().get(), this->data_.byte_size()},
                               this->size());
  }

  // create a const view
  sample_cloud_const_view_t const_view() const {
    return sample_cloud_const_view_t({this->data_.data().pointer().get(), this->data_.byte_size()},
                                     this->size());
  }

  // Allow conversion from owning to mutable view
  template <bool X = kIsMutable>
  operator std::enable_if_t<X, sample_cloud_view_t>() { return view();}
  // Allow conversion to const view
  operator sample_cloud_const_view_t() const { return const_view(); }

  // Resize the sample cloud. This action is destructive
  void resize(size_t number_samples) {
    data_.resize(number_samples, Channels);
    sample_count_ = number_samples;
  }

  // Returns the number of samples in the sample cloud
  size_t size() const { return sample_count_; }
  // Returns the number of data channels in the sample cloud
  constexpr size_t channels() const { return Channels; }

  // Return a mutable eigen map as a matrix to the underlying data
  template <bool X = kIsMutable>
  std::enable_if_t<X, eigen_matrix_view_t> eigen_view() {
    return eigen_matrix_view_t(data_.element_wise_begin(), Channels, sample_count_);
  }

  // Return a const eigen map as a matrix to the underlying data
  eigen_matrix_const_view_t eigen_const_view() const {
    return eigen_matrix_const_view_t(data_.element_wise_begin(), Channels, sample_count_);
  }

  // Get an eigen map view of a vector of data.
  template <bool X = kIsMutable>
  std::enable_if_t<X, eigen_sample_view_t> operator[](size_t index) {
    return eigen_sample_view_t(data_.element_wise_begin() + index * Channels, Channels);
  }

  // Get an constant eigen map view of a vector of data.
  eigen_sample_const_view_t operator[](size_t index) const {
    return eigen_sample_const_view_t(data_.element_wise_begin() + index * Channels, Channels);
  }

  // const access to the underlying buffer object
  const data_t& tensor() const { return data_; }

  // const access to the underlying buffer object
  const buffer_t& data() const { return data_.data(); }
  // access to the underlying buffer object
  template <bool X = kIsMutable>
  std::enable_if_t<X, buffer_t&> data() {
    return data_.data();
  }

 private:
  data_t data_;
  size_t sample_count_ = 0;
};

template <typename K, size_t Channels>
using SampleCloud = SampleCloudBase<K, Channels, CpuBuffer>;

template <typename K, size_t Channels>
using SampleCloudView = SampleCloudBase<K, Channels, CpuBufferView>;

template <typename K, size_t Channels>
using SampleCloudConstView = SampleCloudBase<K, Channels, CpuBufferConstView>;

#define ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, T, S) \
  using SampleCloud##N##S = SampleCloud<T, N>;         \
  using SampleCloudView##N##S = SampleCloudView<T, N>; \
  using SampleCloudConstView##N##S = SampleCloudConstView<T, N>;

#define ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(N)                   \
  template <class K>                                          \
  using SampleCloud##N = SampleCloud<K, N>;                   \
  template <class K>                                          \
  using SampleCloudView##N = SampleCloudView<K, N>;           \
  template <class K>                                          \
  using SampleCloudConstView##N = SampleCloudConstView<K, N>; \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, uint8_t, ub)       \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, uint16_t, ui16)    \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, int, i)            \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, double, d)         \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, float, f)          \
  ISAAC_DECLARE_SAMPLE_CLOUD_TYPES_IMPL(N, float16, f16)

ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(1)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(2)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(3)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(4)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(5)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(6)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(7)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(8)
ISAAC_DECLARE_SAMPLE_CLOUD_TYPES(9)

template <typename K, size_t Channels>
using CudaSampleCloud = SampleCloudBase<K, Channels, CudaBuffer>;

template <typename K, size_t Channels>
using CudaSampleCloudView = SampleCloudBase<K, Channels, CudaBufferView>;

template <typename K, size_t Channels>
using CudaSampleCloudConstView = SampleCloudBase<K, Channels, CudaBufferConstView>;

#define ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, T, S)    \
  using CudaSampleCloud##N##S = CudaSampleCloud<T, N>;         \
  using CudaSampleCloudView##N##S = CudaSampleCloudView<T, N>; \
  using CudaSampleCloudConstView##N##S = CudaSampleCloudConstView<T, N>;

#define ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(N)                      \
  template <class K>                                                  \
  using CudaSampleCloud##N = CudaSampleCloud<K, N>;                   \
  template <class K>                                                  \
  using CudaSampleCloudView##N = CudaSampleCloudView<K, N>;           \
  template <class K>                                                  \
  using CudaSampleCloudConstView##N = CudaSampleCloudConstView<K, N>; \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, uint8_t, ub)          \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, uint16_t, ui16)       \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, int, i)               \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, double, d)            \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, float, f)             \
  ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES_IMPL(N, float16, f16)

ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(1)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(2)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(3)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(4)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(5)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(6)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(7)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(8)
ISAAC_DECLARE_CUDA_SAMPLE_CLOUD_TYPES(9)

}  // namespace isaac
