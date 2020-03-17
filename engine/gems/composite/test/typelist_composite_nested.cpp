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

using MyPartsSub = PartList<Vector2d, Vector3d, double>;

template <typename Container>
struct MyStateSub : TypelistComposite<MyPartsSub, Container> {
  ISAAC_COMPOSITE_BASE(MyPartsSub, Container);
  ISAAC_COMPOSITE_PART_VECTOR(0, tick);
  ISAAC_COMPOSITE_PART_VECTOR(1, trick);
  ISAAC_COMPOSITE_PART_SCALAR(2, track);
};

using MyPartsCombined = PartList<double, Pose2d, CompositeAsPart<MyStateSub>, double>;

template <typename Container>
struct MyStateCombined : TypelistComposite<MyPartsCombined, Container> {
  ISAAC_COMPOSITE_BASE(MyPartsCombined, Container);
  ISAAC_COMPOSITE_PART_SCALAR(0, foo);
  ISAAC_COMPOSITE_PART_POSE2(1, bar);
  ISAAC_COMPOSITE_PART_COMPOSITE(2, tor);
  ISAAC_COMPOSITE_PART_SCALAR(3, zur);
};

TEST(TypelistComposite, NestedComposite) {
  MyStateCombined<EigenVectorCompositeContainer> state;
  EXPECT_EQ(state.data.size(), 12);

  state.foo() = 0.03;
  EXPECT_NEAR(state.foo(), 0.03, 1e-9);

  MyStateSub<EigenVectorCompositeContainer> sub;
  sub.tick() = Vector2d{-1.03, 2.07};
  sub.trick() = Vector3d{-5.05, 8.01, 3.07};
  sub.track() = 0.05;
  state.tor() = sub;

  ISAAC_EXPECT_VEC_NEAR(sub.tick().get(), state.tor().get().tick().get(), 1e-9);

  MyStateSub<EigenVectorCompositeContainer> sub2 = state.tor();
  ISAAC_EXPECT_VEC_NEAR(sub2.tick().get(), sub.tick().get(), 1e-9);

  EXPECT_NEAR(state.scalar<0>(), 0.03, 1e-9);
  EXPECT_NEAR(state.scalar<5>(), -1.03, 1e-9);
  EXPECT_NEAR(state.scalar<6>(), 2.07, 1e-9);
  EXPECT_NEAR(state.scalar<7>(), -5.05, 1e-9);
  EXPECT_NEAR(state.scalar<8>(), 8.01, 1e-9);
  EXPECT_NEAR(state.scalar<9>(), 3.07, 1e-9);
  EXPECT_NEAR(state.scalar<10>(), 0.05, 1e-9);
}

}  // namespace isaac
