/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <algorithm>
#include <utility>

#include "engine/core/assert.hpp"
#include "engine/core/buffers/buffer.hpp"
#include "engine/core/byte.hpp"
#include "engine/core/math/types.hpp"

namespace isaac {

// -------------------------------------------------------------------------------------------------

// Pixel type for image which owns its data
template <typename K, int N>
using Pixel = std::conditional_t<N == 1, K, Vector<K, N>>;

// Pixel type for image which does not own its data but allows changes to it
template <typename K, int N>
using PixelRef = std::conditional_t<N == 1, K&, Eigen::Map<Vector<K, N>>>;

// Pixel type for image which does not own its data and does not allow changes to it
template <typename K, int N>
using PixelConstRef = std::conditional_t<N == 1, const K&, Eigen::Map<const Vector<K, N>>>;

namespace detail {

// Helper type to create pixels and pixel references based on a pointer for N > 1
template <typename K, int N>
struct PixelCreator {
  // Creates an owning pixel based on a pointer
  static Pixel<K, N> CreatePixel(K* ptr) {
    return Pixel<K, N>(ptr);
  }
  // Creates a non-owning pixel based on a pointer
  static PixelRef<K, N> CreatePixelRef(K* ptr) {
    return PixelRef<K, N>(ptr);
  }
  // Creates a non-owning, non-mutable pixel based on a pointer
  static PixelConstRef<K, N> CreatePixelConstRef(const K* ptr) {
    return PixelConstRef<K, N>(ptr);
  }
};

// Specialized version of PixelCreator for N = 1
template <typename K>
struct PixelCreator<K, 1> {
  static Pixel<K, 1> CreatePixel(K* ptr) { return *ptr; }
  static PixelRef<K, 1> CreatePixelRef(K* ptr) { return *ptr; }
  static PixelConstRef<K, 1> CreatePixelConstRef(const K* ptr) { return *ptr; }
};

// Checks if an integer type can be converted to size_t
template <typename IntType>
constexpr bool is_convertible_to_size_t = std::is_convertible<IntType, size_t>::value;

}  // namespace detail

// -------------------------------------------------------------------------------------------------

// Image of type K with N channels.
template <typename K, int N, typename BufferType>
class ImageBase {
 public:
  static_assert(!std::is_const<K>::value, "ImageBase only valid with non-const element type");
  using buffer_t = BufferType;
  static constexpr bool kIsMutable = BufferTraits<buffer_t>::kIsMutable;
  static constexpr bool kIsOwning = BufferTraits<buffer_t>::kIsOwning;

  using element_t = K;
  using element_const_ptr_t = std::add_const_t<K>*;
  using element_ptr_t = std::conditional_t<kIsMutable, K*, element_const_ptr_t>;

  using raw_const_ptr_t = std::add_const_t<byte>*;
  using raw_ptr_t = std::conditional_t<kIsMutable, byte*, raw_const_ptr_t>;

  using pixel_t = Pixel<K, N>;
  using pixel_ref_t = PixelRef<K, N>;
  using pixel_const_ref_t = PixelConstRef<K, N>;

  using buffer_view_t = typename BufferTraits<buffer_t>::buffer_view_t;
  using buffer_const_view_t = typename BufferTraits<buffer_t>::buffer_const_view_t;

  using image_view_t = ImageBase<K, N, buffer_view_t>;
  using image_const_view_t = ImageBase<K, N, buffer_const_view_t>;

  // Create an empty image object
  ImageBase()
      : rows_(0), cols_(0) {}

  // Create an image object of given dimensions and with given storage container
  ImageBase(buffer_t buffer, size_t rows, size_t cols)
  : data_(std::move(buffer)), rows_(rows), cols_(cols) {
    ASSERT(buffer.size() >= getByteSize(), "Buffer too small: %zd vs %zd",
           buffer.size(), getByteSize());
  }

  // Create an image object of given dimensions. This will create a new storage container and is
  // only available for storage container types which own their memory. This function is templated
  // on the integer type so that we can distable this constructor for non-owning storage types.
  template <typename IntType>
  ImageBase(
      // The following enable-if makes sure that this version is only available for owning images.
      std::enable_if_t<kIsOwning && detail::is_convertible_to_size_t<IntType>, IntType> rows,
      IntType cols)
      : rows_(0), cols_(0) {
    resize(rows, cols);
  }
  ImageBase(const Vector2<int>& dimensions) : rows_(0), cols_(0) {
    resize(dimensions);
  }

  // Copy construction uses the default behavior
  ImageBase(const ImageBase& other) = default;
  // Copy assignment uses the default behavior
  ImageBase& operator=(const ImageBase& other) = default;
  // Move construction uses the default behavior
  ImageBase(ImageBase&& other) {
    *this = std::move(other);
  }
  // Move assignment uses the default behavior
  ImageBase& operator=(ImageBase&& other) {
    rows_ = other.rows_;
    cols_ = other.cols_;
    data_ = std::move(other.data_);
    other.rows_ = 0;
    other.cols_ = 0;
    return *this;
  }

  // Create a view.
  image_const_view_t view() const {
    return const_view();
  }
  template <bool X = kIsMutable>
  std::enable_if_t<X, image_view_t> view() {
    return image_view_t({this->data().pointer().get(), this->getByteSize()},
                        this->rows(), this->cols());
  }
  image_const_view_t const_view() const {
    return image_const_view_t({this->data().pointer().get(), this->getByteSize()},
                              this->rows(), this->cols());
  }

  // Allow conversion from owning to mutable view
  template <bool X = kIsMutable>
  operator std::enable_if_t<X && kIsOwning, image_view_t>() {
    return view();
  }
  // Allow conversion to const view for any non const view image (for const view image, the default
  // constructor should be used).
  template <typename X = ImageBase<K, N, BufferType>>
  operator std::enable_if_t<!std::is_same<X, image_const_view_t>::value,
                            image_const_view_t>() const {
    return const_view();
  }

  // Returns true if this image has dimensions 0
  bool empty() const { return rows_ == 0 || cols_ == 0; }
  // Returns the number of rows
  size_t rows() const { return rows_; }
  // Returns the number of cols
  size_t cols() const { return cols_; }
  // (rows, cols) as a 2-vector
  Vector2i dimensions() const { return Vector2i{rows_, cols_}; }
  // The number of channels
  constexpr int channels() const { return N; }
  // The total number of pixels in the image
  size_t num_pixels() const { return rows() * cols(); }

  // Pointers to the first element of the first pixel
  auto element_wise_begin() const {
    return reinterpret_cast<element_const_ptr_t>(data_.pointer().get());
  }
  auto element_wise_begin() {
    return reinterpret_cast<element_ptr_t>(data_.pointer().get());
  }
  // Pointers behind the last element of the last pixel
  auto element_wise_end() const {
    ASSERT(hasTrivialStride(), "element_wise_end() can not be used with non-trivial stride");
    return element_wise_begin() + num_elements();
  }
  auto element_wise_end() {
    ASSERT(hasTrivialStride(), "element_wise_end() can not be used with non-trivial stride");
    return element_wise_begin() + num_elements();
  }
  // The total number of elements in the image
  size_t num_elements() const { return num_pixels() * channels(); }

  // Resizes the image
  template <bool X = kIsOwning>
  std::enable_if_t<X, void> resize(size_t desired_rows, size_t desired_cols) {
    // reallocate only if the new dimensions are different
    if (rows_ != desired_rows || cols_ != desired_cols) {
      rows_ = desired_rows;
      cols_ = desired_cols;
      data_ = buffer_t(getByteSize());
    }
  }
  template <bool X = kIsOwning>
  std::enable_if_t<X, void> resize(const Vector2i& desired) {
    resize(desired[0], desired[1]);
  }

  // Returns true if the given pixel coordinate references a valid pixel.
  bool isValidCoordinate(size_t row, size_t col) const {
    return row < rows_ && col < cols_;
  }

  // Returns a non mutable reference pixel holder on the position (row, col).
  pixel_const_ref_t operator()(size_t row, size_t col) const {
    return detail::PixelCreator<K, N>::CreatePixelConstRef(row_pointer(row) + N*col);
  }
  pixel_const_ref_t operator()(const Vector2i& coordinate) const {
    return detail::PixelCreator<K, N>::CreatePixelConstRef(
        row_pointer(coordinate[0]) + N*coordinate[1]);
  }
  // Returns a mutable reference pixel holder on the position (row, col).
  template <bool X = kIsMutable>
  std::enable_if_t<X, pixel_ref_t> operator()(size_t row, size_t col) {
    return detail::PixelCreator<K, N>::CreatePixelRef(row_pointer(row) + N*col);
  }
  template <bool X = kIsMutable>
  std::enable_if_t<X, pixel_ref_t> operator()(const Vector2i& coordinate) {
    return detail::PixelCreator<K, N>::CreatePixelRef(
        row_pointer(coordinate[0]) + N*coordinate[1]);
  }

  // Returns a non mutable reference pixel holder at the position `index`.
  pixel_const_ref_t operator[](size_t index) const {
    // FIXME The following code does not work with non-trivial stride
    return detail::PixelCreator<K, N>::CreatePixelConstRef(element_wise_begin() + N*index);
  }
  // Returns a mutable reference pixel holder at the position `index`.
  template <bool X = kIsMutable>
  std::enable_if_t<X, pixel_ref_t> operator[](size_t index) {
    // FIXME The following code does not work with non-trivial stride
    return detail::PixelCreator<K, N>::CreatePixelRef(element_wise_begin() + N*index);
  }

  // Pointer to the first element in the i-th row in an image
  element_ptr_t row_pointer(size_t row) {
    return reinterpret_cast<element_ptr_t>(
        reinterpret_cast<raw_ptr_t>(element_wise_begin()) + row * getStride());
  }
  element_const_ptr_t row_pointer(size_t row) const {
    return reinterpret_cast<element_const_ptr_t>(
        reinterpret_cast<raw_const_ptr_t>(element_wise_begin()) + row * getStride());
  }

  // Total number of bytes used to store the image.
  size_t getByteSize() const { return rows() * getUsedBytesPerRow(); }
  // Number of bytes used to store one row in memory. This might be more than the number of bytes
  // which are actually required to store a full row of pixel data. A non-trivial `stride` may
  // increase performance and is for example used by default in CUDA.
  size_t getStride() const { return getUsedBytesPerRow(); }
  // Number of bytes used per row to store data. This might be smaller than getStride().
  size_t getUsedBytesPerRow() const { return cols() * channels() * sizeof(K); }
  // Returns true if the image has a trivial stride, i.e. no extra padding at the end of each row
  bool hasTrivialStride() const { return getStride() == getUsedBytesPerRow(); }

  // The underlying buffer object
  const buffer_t& data() const { return data_; }
  buffer_t& data() { return data_; }

 private:
  buffer_t data_;

  size_t rows_;
  size_t cols_;
};

// -------------------------------------------------------------------------------------------------

// An image which owns it's memory
template <class K, int N>
using Image = ImageBase<K, N, CpuBuffer>;

// A mutable view on an image which does not own memory but can be used to read and write the
// data of the underlying image.
template <class K, int N>
using ImageView = ImageBase<K, N, CpuBufferView>;

// A non-mutable view on an image which does not own the memory and can only be used to read
// the data of the underlying image.
template <class K, int N>
using ImageConstView = ImageBase<K, N, CpuBufferConstView>;

#define ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, T, S)       \
  using Image##N##S          = Image<T, N>;           \
  using ImageView##N##S      = ImageView<T, N>;       \
  using ImageConstView##N##S = ImageConstView<T, N>;  \
  using Pixel##N##S          = Pixel<T, N>;           \
  using PixelRef##N##S       = PixelRef<T, N>;        \
  using PixelConstRef##N##S  = PixelConstRef<T, N>;   \

#define ISAAC_DECLARE_IMAGE_TYPES(N)                                  \
  template <class K> using Image##N = Image<K, N>;                    \
  template <class K> using ImageView##N = ImageView<K, N>;            \
  template <class K> using ImageConstView##N = ImageConstView<K, N>;  \
  ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, uint8_t,  ub)    \
  ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, uint16_t, ui16)  \
  ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, int,      i)     \
  ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, double,   d)     \
  ISAAC_DECLARE_IMAGE_TYPES_IMPL(N, float,    f)     \

ISAAC_DECLARE_IMAGE_TYPES(1)
ISAAC_DECLARE_IMAGE_TYPES(2)
ISAAC_DECLARE_IMAGE_TYPES(3)
ISAAC_DECLARE_IMAGE_TYPES(4)

// -------------------------------------------------------------------------------------------------

// Helper function to create an image view from a pointer using pitched storage
template <class K, int N>
ImageView<K, N> CreateImageView(K* data, size_t rows, size_t cols, size_t stride) {
  return ImageView<K, N>(
      CpuBufferView(reinterpret_cast<const byte*>(data), rows * cols * N * sizeof(K)),
      rows, cols);
}

// Helper function to create an image const view from a pointer using dense storage
template <class K, int N>
ImageConstView<K, N> CreateImageView(const K* data, size_t rows, size_t cols) {
  return ImageConstView<K, N>(
      CpuBufferConstView(reinterpret_cast<const byte*>(data), rows * cols * N * sizeof(K)),
      rows, cols);
}

// -------------------------------------------------------------------------------------------------

// An image stored in device memory which owns it's memory
template <class K, int N>
using CudaImage = ImageBase<K, N, CudaBuffer>;

// A mutable view on an image which is stored on GPU device memory, does not own memory, but can be
// used to read and write the data of the underlying image.
template <class K, int N>
using CudaImageView = ImageBase<K, N, CudaBufferView>;

// A non-mutable view on an image which is stored on GPU device memory, does not own its memory, and
// can only be used to read the data of the underlying image.
template <class K, int N>
using CudaImageConstView = ImageBase<K, N, CudaBufferConstView>;

// Same as CudaImage except using trivial pitch
template <class K, int N>
using CudaContinuousImage = ImageBase<K, N, CudaBuffer>;
// Same as CudaImageView except using trivial pitch
template <class K, int N>
using CudaContinuousImageView = ImageBase<K, N, CudaBufferView>;
// Same as CudaImageConstView except using trivial pitch
template <class K, int N>
using CudaContinuousImageConstView = ImageBase<K, N, CudaBufferConstView>;

// Helper macro for ISAAC_DECLARE_CUDA_IMAGE_TYPES
#define ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, K, S)          \
  using CudaImage##N##S          = CudaImage<K, N>;           \
  using CudaImageView##N##S      = CudaImageView<K, N>;       \
  using CudaImageConstView##N##S = CudaImageConstView<K, N>;  \

// Helper macro to define various CudaImage types
#define ISAAC_DECLARE_CUDA_IMAGE_TYPES(N)                                     \
  template <class K> using CudaImage##N          = CudaImage<K, N>;           \
  template <class K> using CudaImageView##N      = CudaImageView<K, N>;       \
  template <class K> using CudaImageConstView##N = CudaImageConstView<K, N>;  \
  ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, uint8_t,  ub)    \
  ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, uint16_t, ui16)  \
  ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, int,      i)     \
  ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, double,   d)     \
  ISAAC_DECLARE_CUDA_IMAGE_TYPES_IMPL(N, float,    f)     \

ISAAC_DECLARE_CUDA_IMAGE_TYPES(1)
ISAAC_DECLARE_CUDA_IMAGE_TYPES(2)
ISAAC_DECLARE_CUDA_IMAGE_TYPES(3)
ISAAC_DECLARE_CUDA_IMAGE_TYPES(4)

// Macros for declaring types for CudaContinuousImage.
#define ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, K, S)          \
  using CudaContinuousImage##N##S          = CudaContinuousImage<K, N>;           \
  using CudaContinuousImageView##N##S      = CudaContinuousImageView<K, N>;       \
  using CudaContinuousImageConstView##N##S = CudaContinuousImageConstView<K, N>;  \

#define ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES(N)                                     \
  template <class K> using CudaContinuousImage##N          = CudaContinuousImage<K, N>;           \
  template <class K> using CudaContinuousImageView##N      = CudaContinuousImageView<K, N>;       \
  template <class K> using CudaContinuousImageConstView##N = CudaContinuousImageConstView<K, N>;  \
  ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, uint8_t,  ub)    \
  ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, uint16_t, ui16)  \
  ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, int,      i)     \
  ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, double,   d)     \
  ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES_IMPL(N, float,    f)     \

ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES(1)
ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES(2)
ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES(3)
ISAAC_DECLARE_CUDA_CONTINUOUS_IMAGE_TYPES(4)

// -------------------------------------------------------------------------------------------------

}  // namespace isaac
