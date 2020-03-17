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
#include "engine/gems/composite/containers/memory_view.hpp"
#include "engine/gems/composite/containers/tuple.hpp"
#include "engine/gems/math/test_utils.hpp"
#include "gtest/gtest.h"

namespace isaac {

using MyParts = PartList<double, Pose2d, double, Vector2d>;

template <typename ContainerT>
struct MyState : TypelistComposite<MyParts, ContainerT> {
  ISAAC_COMPOSITE_BASE(MyParts, ContainerT);
  ISAAC_COMPOSITE_PART_SCALAR(0, foo);
  ISAAC_COMPOSITE_PART_POSE2(1, bar);
  ISAAC_COMPOSITE_PART_SCALAR(2, zur);
  ISAAC_COMPOSITE_PART_VECTOR(3, tor);
};

TEST(TypelistComposite, ElementStartIndex) {
  EXPECT_EQ(Length<MyParts>::value, 8);
  EXPECT_EQ((ElementStartIndex<0, MyParts>::value), 0);
  EXPECT_EQ((ElementStartIndex<1, MyParts>::value), 1);
  EXPECT_EQ((ElementStartIndex<2, MyParts>::value), 5);
  EXPECT_EQ((ElementStartIndex<3, MyParts>::value), 6);
}

template <typename ContainerT>
void MyStateTestWriteFuncReadFunc(MyState<ContainerT>& state) {
  state.foo() = 1.1;
  EXPECT_NEAR(state.foo(), 1.1, 1e-9);

  const Pose2d expected{SO2d::FromAngle(1.0), Vector2d{-1.2, 0.7}};
  state.bar() = expected;
  const Pose2d actual = state.bar();
  ISAAC_EXPECT_POSE_NEAR(actual, expected, 1e-9);

  state.zur() = -3.9;
  EXPECT_NEAR(state.zur(), -3.9, 1e-9);

  state.tor() = Vector2d{0.27, 0.54};
  ISAAC_EXPECT_VEC_NEAR(Evaluate(state.tor()), Vector2d(0.27, 0.54), 1e-9);
}

template <typename ContainerT>
void MyStateTestWriteFuncReadPart(MyState<ContainerT>& state) {
  state.foo() = 1.31;
  EXPECT_NEAR(state.template part<0>(), 1.31, 1e-9);

  const Pose2d expected{SO2d::FromAngle(1.40), Vector2d{-1.92, 0.57}};
  state.bar() = expected;
  const Pose2d actual = state.template part<1>();
  ISAAC_EXPECT_POSE_NEAR(actual, expected, 1e-9);

  state.zur() = -3.79;
  EXPECT_NEAR(state.template part<2>(), -3.79, 1e-9);

  state.tor() = Vector2d{0.278, 0.549};
  ISAAC_EXPECT_VEC_NEAR(Evaluate(state.template part<3>()), Vector2d(0.278, 0.549), 1e-9);
}

template <typename ContainerT>
void MyStateTestWritePartReadFunc(MyState<ContainerT>& state) {
  state.template part<0>() = 1.431;
  EXPECT_NEAR(state.foo(), 1.431, 1e-9);

  const Pose2d expected{SO2d::FromAngle(1.640), Vector2d{-1.292, 0.057}};
  state.template part<1>() = expected;
  const Pose2d actual = state.bar();
  ISAAC_EXPECT_POSE_NEAR(actual, expected, 1e-9);

  state.template part<2>() = -3.679;
  EXPECT_NEAR(state.zur(), -3.679, 1e-9);

  state.template part<3>() = Vector2d{0.2782, 0.5493};
  ISAAC_EXPECT_VEC_NEAR(Evaluate(state.tor()), Vector2d(0.2782, 0.5493), 1e-9);
}

template <typename ContainerT>
void MyStateTestWriteFuncReadScalar(MyState<ContainerT>& state) {
  state.foo() = 1.51;
  EXPECT_NEAR(state.template scalar<0>(), 1.51, 1e-9);

  const Pose2d expected{SO2d::FromAngle(1.50), Vector2d{-1.52, 0.57}};
  state.bar() = expected;
  EXPECT_NEAR(state.template scalar<1>(), expected.rotation.cos(), 1e-9);
  EXPECT_NEAR(state.template scalar<2>(), expected.rotation.sin(), 1e-9);
  EXPECT_NEAR(state.template scalar<3>(), expected.translation.x(), 1e-9);
  EXPECT_NEAR(state.template scalar<4>(), expected.translation.y(), 1e-9);

  state.zur() = -3.59;
  EXPECT_NEAR(state.template scalar<5>(), -3.59, 1e-9);

  state.tor() = Vector2d{0.1278, 0.2549};
  EXPECT_NEAR(state.template scalar<6>(), 0.1278, 1e-9);
  EXPECT_NEAR(state.template scalar<7>(), 0.2549, 1e-9);
}

template <typename ContainerT>
void MyStateTestWriteElementReadFunc(MyState<ContainerT>& state) {
  state.template scalar<0>() = 1.51;
  EXPECT_NEAR(state.foo(), 1.51, 1e-9);

  const Pose2d expected{SO2d::FromAngle(1.50), Vector2d{-1.52, 0.57}};

  // We can not write to sin and cos of an SO2 object directly. The following code does the
  // trick for the purpose of this test, but it is not ideal yet.
  // state.template scalar<1>() = expected.rotation.sin();  // Note: disabled
  // state.template scalar<2>() = expected.rotation.cos();  // Note: disabled
  const Pose2d expected_wrong{expected.rotation, 100.0*expected.translation};  // wrong translation
  state.template part<1>() = expected_wrong;

  state.template scalar<3>() = expected.translation.x();
  state.template scalar<4>() = expected.translation.y();
  const Pose2d actual = state.bar();
  ISAAC_EXPECT_POSE_NEAR(actual, expected, 1e-9);

  state.template scalar<5>() = -3.59;
  EXPECT_NEAR(state.zur(), -3.59, 1e-9);

  state.template scalar<6>() = 0.27825;
  state.template scalar<7>() = 0.54935;
  ISAAC_EXPECT_VEC_NEAR(Evaluate(state.tor()), Vector2d(0.27825, 0.54935), 1e-9);
}

template <typename ContainerT>
void MyStateTest(MyState<ContainerT>& state) {
  MyStateTestWriteFuncReadFunc(state);
  MyStateTestWriteFuncReadPart(state);
  MyStateTestWritePartReadFunc(state);
  MyStateTestWriteFuncReadScalar(state);
  MyStateTestWriteElementReadFunc(state);
}

TEST(TypelistComposite, EigenContainer) {
  MyState<EigenVectorCompositeContainer> state;
  MyStateTest(state);
}

TEST(TypelistComposite, Names) {
  MyState<EigenVectorCompositeContainer> state;
  EXPECT_EQ(state.part_name<0>(), "foo");
  EXPECT_EQ(state.part_name<1>(), "bar");
  EXPECT_EQ(state.part_name<2>(), "zur");
  EXPECT_EQ(state.scalar_name<0>(), "foo");
  EXPECT_EQ(state.scalar_name<1>(), "bar/px");
  EXPECT_EQ(state.scalar_name<2>(), "bar/py");
  EXPECT_EQ(state.scalar_name<3>(), "bar/qx");
  EXPECT_EQ(state.scalar_name<4>(), "bar/qy");
  EXPECT_EQ(state.scalar_name<5>(), "zur");
}

TEST(TypelistComposite, PointerContainer) {
  std::vector<double> buffer(Length<MyParts>::value);
  MyState<MemoryViewCompositeContainer> state;
  state.data = CpuArrayView<double>(buffer.data(), buffer.size());
  MyStateTest(state);
}

TEST(TypelistComposite, TupleContainer) {
  MyState<TupleCompositeContainer> state;
  MyStateTestWriteFuncReadFunc(state);
  MyStateTestWriteFuncReadPart(state);
  MyStateTestWritePartReadFunc(state);
  // TODO: Compile-time scalar access for tuple container not yet supported
  // MyStateTestWriteFuncReadScalar(state);
  // MyStateTestWriteElementReadFunc(state);
}

}  // namespace isaac
