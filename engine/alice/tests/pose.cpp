/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyPoseTest : public Codelet {
 public:

  void start() override {
    set_world_T_robot(Pose2d{SO2d::FromAngle(DegToRad(60.0)), Vector2d{3.0, -2.1}},
                      getTickTime());
    tickPeriodically();
    stopwatch().start();
  }
  void tick() override {
    Pose2d world_T_robot = get_world_T_robot(getTickTime());
    world_T_robot.translation += world_T_robot.rotation * Vector2d{get_speed()*getTickDt(), 0.0};
    set_world_T_robot(world_T_robot, getTickTime());
    const double expected = get_speed() * stopwatch().read();
    const double distance = (get_world_T_robot(getTickTime()).translation - Vector2d{3.0, -2.1}).norm();
    EXPECT_NEAR(distance, expected, 0.05);
  }
  void stop() override {
  }

  ISAAC_PARAM(double, speed, 1.7)
  ISAAC_POSE2(world, robot)
  private:
};

TEST(Pose, Hooks) {
  Application app;
  Node* node = app.createNode("test");
  auto* poseTest = node->addComponent<MyPoseTest>();
  poseTest->async_set_tick_period("100ms");
  app.startWaitStop(0.50);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyPoseTest);
