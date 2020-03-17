/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "KinovaJacoSampleController.hpp"

#include "messages/messages.hpp"
#include "messages/state/kinova_jaco.hpp"

namespace isaac {
namespace kinova_jaco {

void KinovaJacoSampleController::start() {
  tickPeriodically();
}

void KinovaJacoSampleController::tick() {
  // Set home pose
  messages::JacoCartesianPose cartesian_pose;
  const Pose3d home_pose = get_home_pose();
  const Vector3d& home_position = home_pose.translation;
  cartesian_pose.px() = home_position.x();
  cartesian_pose.py() = home_position.y();
  cartesian_pose.pz() = home_position.x();
  cartesian_pose.setOrientationFromQuaternion(home_pose.rotation.quaternion());

  // Serialize and publish cartesian pose command
  ToProto(cartesian_pose, tx_cartesian_pose_command().initProto(),
          tx_cartesian_pose_command().buffers());
  tx_cartesian_pose_command().publish();
}

}  // namespace kinova_jaco
}  // namespace isaac
