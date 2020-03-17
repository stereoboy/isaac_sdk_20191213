/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/gems/composite/typelist_composite.hpp"

#include "engine/gems/composite/containers/eigen.hpp"
#include "engine/gems/math/test_utils.hpp"
#include "gtest/gtest.h"

namespace isaac {

using MyParts = PartList<double, Pose2d, double>;

struct MyState : TypelistComposite<MyParts, EigenVectorCompositeContainer> {
  ISAAC_COMPOSITE_BASE(MyParts, EigenVectorCompositeContainer);
  ISAAC_COMPOSITE_PART_SCALAR(0, foo);
  ISAAC_COMPOSITE_PART_POSE2(1, bar);
  ISAAC_COMPOSITE_PART_SCALAR(2, zur);
};

TEST(TypelistComposite, Basics) {
  constexpr double kValue1a = 0.03;
  constexpr double kValue1b = 1.31;
  constexpr double kValue2a = 1.40;
  constexpr double kValue2b = -1.92;
  constexpr double kValue2c = 0.57;
  constexpr double kValue3 = -3.79;

  MyState state;

  // Check simple read/write
  state.foo() = kValue1a;
  EXPECT_EQ(state.foo(), kValue1a);
  state.foo() = kValue1b;
  EXPECT_EQ(state.part<0>(), kValue1b);

  // Check read/write for more complicated type
  const Pose2d expected{SO2d::FromAngle(kValue2a), Vector2d{kValue2b, kValue2c}};
  state.bar() = expected;
  const Pose2d actual = state.part<1>();
  ISAAC_EXPECT_POSE_NEAR(actual, expected, 1e-9);

  // Read/write a scalar at the end
  state.zur() = kValue3;
  EXPECT_EQ(state.part<2>(), kValue3);

  // Check that the data is represented as a vector in the expected order
  Vector<double, 6> expected_vec;
  expected_vec[0] = kValue1b;
  expected_vec[1] = std::cos(kValue2a);
  expected_vec[2] = std::sin(kValue2a);
  expected_vec[3] = kValue2b;
  expected_vec[4] = kValue2c;
  expected_vec[5] = kValue3;
  ISAAC_EXPECT_VEC_EQ(state.data, expected_vec);
}

}  // namespace isaac
