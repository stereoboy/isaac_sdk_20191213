/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/alice_codelet.hpp"
#include "engine/gems/state/io.hpp"

namespace isaac {
namespace kinova_jaco {

// A sample controller for joint velocities for the Jaco arm.
// Arm should be in the home position before running this sample controller.
// If the arm is in another position self-collisions could occur, causing damage
// to the arm or user.
class KinovaJacoSampleJointVelocityController : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override {}

  // Command for joint velocity control
  ISAAC_PROTO_TX(StateProto, joint_velocity_command);

 private:
  // Start time initialized in start()
  double start_time_seconds_;
};

}  // namespace kinova_jaco
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::kinova_jaco::KinovaJacoSampleJointVelocityController);
