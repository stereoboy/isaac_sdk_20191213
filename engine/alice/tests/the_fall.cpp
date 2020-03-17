/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

namespace {

constexpr double kGravitationalAcceleration = 9.80665;

}  // namespace

// Let's an object fall in space for a while tracking its position with a pose component.
class TheFall : public Codelet {
 public:
  void start() override {
    tickPeriodically();
    set_world_T_robot(Pose3d::Identity(), getTickTime());
    velocity_ = Vector3d::Zero();
    stopwatch().start();
  }
  void tick() override {
    const double dt = getTickDt();
    Pose3d world_T_robot = get_world_T_robot(getTickTime());
    velocity_ += dt * Vector3d{0, 0, -kGravitationalAcceleration};
    world_T_robot.translation += dt * velocity_;
    set_world_T_robot(world_T_robot, getTickTime());
    // Check if our integration using the pose graph matches physics
    const double time = stopwatch().read();
    const double expected = -0.5 * time * time * kGravitationalAcceleration;
    EXPECT_NEAR(world_T_robot.translation.z(), expected, 0.05);
  }
  ISAAC_POSE3(world, robot)

 private:
  Vector3d velocity_;
};

TEST(Alice, ClockTicking) {
  Application app;
  Node* node = app.createNode("a");
  auto* fall = node->addComponent<TheFall>();
  fall->async_set_tick_period("10ms");
  app.startWaitStop(0.35);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::TheFall);
