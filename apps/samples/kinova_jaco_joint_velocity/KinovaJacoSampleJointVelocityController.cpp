/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "KinovaJacoSampleJointVelocityController.hpp"

#include "messages/messages.hpp"
#include "messages/state/kinova_jaco.hpp"

namespace isaac {
namespace kinova_jaco {

void KinovaJacoSampleJointVelocityController::start() {
  start_time_seconds_ = getTickTime();
  tickPeriodically();
}

void KinovaJacoSampleJointVelocityController::tick() {
  // Elapsed seconds since start of codelet
  const double elapsed_seconds = getTickTime() - start_time_seconds_;

  messages::JacoJointVelocity joint_velocity;
  const double wave = std::sin(0.5 * elapsed_seconds);
  joint_velocity.actuator_1() = 0.5 * wave;
  joint_velocity.actuator_2() = 0.0;
  joint_velocity.actuator_3() = 0.0;
  joint_velocity.actuator_4() = 0.0;
  joint_velocity.actuator_5() = 0.0;
  joint_velocity.actuator_6() = -0.5 * wave;
  joint_velocity.actuator_7() = 0.5 * wave;

  // Serialize and publish joint velocity command
  ToProto(joint_velocity, tx_joint_velocity_command().initProto(),
          tx_joint_velocity_command().buffers());
  tx_joint_velocity_command().publish();
}

}  // namespace kinova_jaco
}  // namespace isaac
