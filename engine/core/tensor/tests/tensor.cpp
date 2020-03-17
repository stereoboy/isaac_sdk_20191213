/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/core/tensor/tensor.hpp"

#include <vector>

#include "engine/gems/math/float16.hpp"
#include "gtest/gtest.h"

namespace isaac {

TEST(Tensor, Order) {
  Tensor1d t1(13);
  EXPECT_EQ(t1.order(), 1);
  EXPECT_EQ(t1.dimensions().size(), 1);
  EXPECT_EQ(t1.dimensions()[0], 13);

  Tensor2d t2(13, 17);
  EXPECT_EQ(t2.order(), 2);
  EXPECT_EQ(t2.dimensions().size(), 2);
  EXPECT_EQ(t2.dimensions()[0], 13);
  EXPECT_EQ(t2.dimensions()[1], 17);

  Tensor3d t3(13, 17, 19);
  EXPECT_EQ(t3.order(), 3);
  EXPECT_EQ(t3.dimensions().size(), 3);
  EXPECT_EQ(t3.dimensions()[0], 13);
  EXPECT_EQ(t3.dimensions()[1], 17);
  EXPECT_EQ(t3.dimensions()[2], 19);

  Tensor4d t4(13, 17, 19, 23);
  EXPECT_EQ(t4.order(), 4);
  EXPECT_EQ(t4.dimensions().size(), 4);
  EXPECT_EQ(t4.dimensions()[0], 13);
  EXPECT_EQ(t4.dimensions()[1], 17);
  EXPECT_EQ(t4.dimensions()[2], 19);
  EXPECT_EQ(t4.dimensions()[3], 23);

  Tensor<double, 14> t14;
  EXPECT_EQ(t14.order(), 14);
  EXPECT_EQ(t14.dimensions().size(), 14);
}

TEST(Tensor, MemoryAccess) {
  {
    // default row order test
    Tensor4i t(2, 3, 4, 5);
    ASSERT_EQ(t.dimensions().size(), 4);
    ASSERT_EQ(t.dimensions()[0], 2);
    ASSERT_EQ(t.dimensions()[1], 3);
    ASSERT_EQ(t.dimensions()[2], 4);
    ASSERT_EQ(t.dimensions()[3], 5);
    ASSERT_EQ(t.data().size(), 4*5*4*3*2);

    int count = 0;
    for (size_t i = 0; i < 2; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        for (size_t k = 0; k < 4; ++k) {
          for (size_t l = 0; l < 5; ++l) {
            t(i, j, k, l) = ++count;
          }
        }
      }
    }

    t(0, 0, 0, 4) = 2;
    t(1, 2, 3, 4) = 15;
    t(1, 1, 3, 1) = 17;

    const Vector4i shifts{3*4*5, 4*5, 5, 1};

    EXPECT_EQ(*(t.element_wise_begin() + shifts.dot(Vector4i{0, 0, 0, 4})), 2);
    EXPECT_EQ(*(t.element_wise_begin() + shifts.dot(Vector4i{1, 2, 3, 4})), 15);
    EXPECT_EQ(*(t.element_wise_begin() + shifts.dot(Vector4i{1, 1, 3, 1})), 17);
  }
}

TEST(Tensor, Allocation) {
  Tensor3d t3;
  Tensor1d t1;
  t1 = Tensor1d(15);
  t3 = Tensor3d(30, 60, 15);
  auto dim3d = t3.dimensions();
  auto dim1d = t1.dimensions();
  EXPECT_EQ(dim3d[0], 30);
  EXPECT_EQ(dim3d[1], 60);
  EXPECT_EQ(dim3d[2], 15);
  EXPECT_EQ(dim1d[0], 15);
}

TEST(Tensor, Validation) {
  Tensor1d t1;
  t1 = Tensor1d(0);
  auto dim1d = t1.dimensions();
  EXPECT_EQ(dim1d[0], 0);
}

TEST(Tensor, Views) {
  Tensor4i t(2, 3, 4, 5);
  t(0, 0, 0, 4) = 2;
  t(1, 2, 3, 4) = 15;

  TensorView4i v = t.view();
  EXPECT_EQ(v(0, 0, 0, 4), 2);
  EXPECT_EQ(v(1, 2, 3, 4), 15);

  v(0, 0, 0, 4) = 22;
  EXPECT_EQ(v(0, 0, 0, 4), 22);
}

TEST(Tensor, ConstViews) {
  Tensor4i t(2, 3, 4, 5);
  t(0, 0, 0, 4) = 2;
  t(1, 2, 3, 4) = 15;

  TensorConstView4i v = t.const_view();

  EXPECT_EQ(v(0, 0, 0, 4), 2);
  EXPECT_EQ(v(1, 2, 3, 4), 15);
}

TEST(Tensor, ConstAccess) {
  Tensor3d t3(20, 40, 59);
  Tensor1d t1(30);
  const Tensor1d& tref1 = t1;
  const Tensor3d& tref3 = t3;
  t3(3, 5, 1) = 12.0;
  EXPECT_EQ(tref3(3, 5, 1), 12.0);
  EXPECT_EQ(tref3(3, 5, 1), 12.0);

  t1(3) = 12.0;
  EXPECT_EQ(tref1(3), 12.0);
}

TEST(Tensor, ConvertViewToConstView) {
  Vector3<size_t> dims = {12, 13, 14};
  Tensor3ub t(dims);
  TensorView3ub view1 = t.view();
  TensorConstView3ub view2 = view1;
  EXPECT_EQ(view2.dimensions(), dims);
}

TEST(Tensor, EmptyView) {
  Vector3<size_t> dims = {0, 0, 0};
  TensorView3ub view;
  EXPECT_EQ(view.element_count(), 0);
  EXPECT_EQ(dims, view.dimensions());
}

TEST(Tensor, EmptyConstView) {
  Vector3<size_t> dims = {0, 0, 0};
  TensorConstView3ub view;
  EXPECT_EQ(view.element_count(), 0);
  EXPECT_EQ(dims, view.dimensions());
}

TEST(Tensor, CopyView) {
  Vector3<size_t> dims = {12, 13, 14};
  Tensor3ub t(dims);
  TensorView3ub view1 = t.view();
  TensorView3ub view2;
  view2 = view1;
  EXPECT_NE(view2.element_count(), 0);
  EXPECT_EQ(view2.dimensions(), dims);
}

TEST(Tensor, CopyConstView) {
  Vector3<size_t> dims = {12, 13, 14};
  Tensor3ub t(dims);
  TensorConstView3ub view1 = t.const_view();
  TensorConstView3ub view2;
  view2 = view1;
  EXPECT_NE(view2.element_count(), 0);
  EXPECT_EQ(view2.dimensions(), dims);
}

TEST(Tensor, Tensor1f16) {
  Tensor<float16, 2> t(20, 40);
  for (size_t i = 0; i < t.dimensions()[0]; i++) {
    for (size_t j = 0; j < t.dimensions()[1]; j++) {
      t(i, j) = static_cast<float16>(i * j);
    }
  }

  for (size_t i = 0; i < t.dimensions()[0]; i++) {
    for (size_t j = 0; j < t.dimensions()[1]; j++) {
      t(i, j) += 3.1;
    }
  }

  for (size_t i = 0; i < t.dimensions()[0]; i++) {
    for (size_t j = 0; j < t.dimensions()[1]; j++) {
      EXPECT_NEAR(t(i, j), 3.1 + static_cast<double>(i * j), 0.15);
    }
  }
}

namespace {

void FooReadable(const TensorConstView1ub& view) {}

void FooWriteable(const TensorView1ub& view) {}

template <size_t N, typename Container>
void FooTemplated(const TensorBase<uint8_t, N, Container>& view) {}

}  // namespace

TEST(Tensor, CallFunctionReadable) {
  Tensor1ub t(10);
  FooReadable(t);
  FooReadable(t.view());
  FooReadable(t.const_view());
}

TEST(Tensor, CallFunctionWriteable) {
  Tensor1ub t(10);
  FooWriteable(t);
  FooWriteable(t.view());
  // FooWriteable(t.const_view());  // should not compile
}

TEST(Tensor, CallFunctionTemplated) {
  Tensor1ub t1(10);
  FooTemplated(t1);
  FooTemplated(t1.view());
  FooTemplated(t1.const_view());
  Tensor3ub t3(10, 20, 3);
  FooTemplated(t3);
  FooTemplated(t3.view());
  FooTemplated(t3.const_view());
}

TEST(Tensor, ResizeReallocatesIfNewDimsAreDifferent) {
  Tensor3ub t(10, 15, 3);
  EXPECT_EQ(t.element_count(), 10 * 15 * 3);
  t.resize(11, 20, 3);
  EXPECT_EQ(t.element_count(), 11 * 20 * 3);
  t.resize(10, 15, 3);
  EXPECT_EQ(t.element_count(), 10 * 15 * 3);
}

TEST(Tensor, Slice) {
  Tensor3i tensor(9, 7, 3);
  for (size_t i0 = 0; i0 < tensor.dimensions()[0]; i0++) {
    for (size_t i1 = 0; i1 < tensor.dimensions()[1]; i1++) {
      for (size_t i2 = 0; i2 < tensor.dimensions()[2]; i2++) {
        tensor(i0, i1, i2) = i0 * 10000 + i1 * 100 + i2;
      }
    }
  }
  for (size_t i0 = 0; i0 < tensor.dimensions()[0]; i0++) {
    TensorConstView2i slice = tensor.slice(i0);
    ASSERT_EQ(tensor.dimensions()[1], slice.dimensions()[0]);
    ASSERT_EQ(tensor.dimensions()[2], slice.dimensions()[1]);
    for (size_t i1 = 0; i1 < tensor.dimensions()[0]; i1++) {
      for (size_t i2 = 0; i2 < tensor.dimensions()[1]; i2++) {
        ASSERT_EQ(slice(i1, i2), tensor(i0, i1, i2));
      }
    }
  }
}

}  // namespace isaac
