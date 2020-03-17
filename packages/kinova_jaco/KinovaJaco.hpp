/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>

#include "engine/alice/alice_codelet.hpp"
#include "engine/alice/message.hpp"
#include "engine/core/optional.hpp"
#include "KinovaTypes.h"
#include "messages/messages.hpp"
#include "messages/state/kinova_jaco.hpp"
#include "packages/kinova_jaco/gems/kinova_jaco_api.hpp"

namespace isaac {
namespace kinova_jaco {

// Available control modes for the Kinova Jaco arm. Control mode restricts
// the types of control commands that can be parsed by the driver in
// packages/kinova_jaco/KinovaJaco.cpp
enum ControlMode {
    kCartesianPose,  // Control position and orientation for end effector
    kJointVelocity,  // Control instantaneous velocity for each joint
    kInvalid = -1    // Invalid control mode, returned from an invalid string in JSON
};

// Mapping between each ControlMode type and an identifying string.
// See nlohmann/json for detailed documentation:
// https://github.com/nlohmann/json#specializing-enum-conversion
NLOHMANN_JSON_SERIALIZE_ENUM(ControlMode, {
  { kInvalid, nullptr },                 // Returned if JSON string does not match another mode
  { kCartesianPose, "cartesian pose" },  // Control position and orientation for end effector
  { kJointVelocity, "joint velocity" }   // Control instantaneous velocity for each joint
});

// A class to receive command and publish state information for the Kinova Jaco arm.
class KinovaJaco : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;

  // Command for end effector position and orientation
  ISAAC_PROTO_RX(StateProto, cartesian_pose_command);

  // Command for angular velocities for joints
  ISAAC_PROTO_RX(StateProto, joint_velocity_command);

  // Current position and orientation of end effector
  ISAAC_PROTO_TX(StateProto, cartesian_pose);

  // Current angle, in Radians, for each joint (7-dof)
  ISAAC_PROTO_TX(StateProto, joint_position);

  // Current angular velocity, in Radians/sec, for each joint (7-dof)
  ISAAC_PROTO_TX(StateProto, joint_velocity);

  // Current position for each finger
  ISAAC_PROTO_TX(StateProto, finger_position);

  // Path to JacoSDK is set in jaco_driver_config.json.
  // Driver is tested for use with JACO2SDK v1.4.2
  // Jaco SDK source: https://drive.google.com/file/d/17_jLW5EWX9j3aY3NGiBps7r77U2L64S_/view
  ISAAC_PARAM(std::string, kinova_jaco_sdk_path);

  // Set control mode for arm. Can only accept commands corresponding to the current mode.
  ISAAC_PARAM(ControlMode, control_mode, kCartesianPose);

 private:
  // Moves the arm to the ready position and initializes the fingers
  void resetArm();

  // Parse cartesian position command and moves end effector to specified position and orientation
  void parseCartesianPoseCommand();

  // Parse joint velocity command and sets each joint to specified velocity
  void parseJointVelocityCommand();

  // Initialize trajectory_point with no limitations active and no hand movement
  void initializeTrajectoryPoint(TrajectoryPoint& trajectory_point);

  // Set trajectory_point to cartesian pose mode
  void initializeCartesianTrajectoryPoint(TrajectoryPoint& trajectory_point);

  // Set trajectory_point to angular velocity mode
  void initializeJointVelocityTrajectoryPoint(TrajectoryPoint& trajectory_point);

  // Assign commanded position to trajectory_point
  void assignCartesianPositionToArm(const messages::JacoCartesianPose& jaco_cartesian_pose,
                                    TrajectoryPoint& trajectory_point);

  // Assign commanded orientation to trajectory_point (converting from quaternion to
  // RPY Euler angles)
  void assignOrientationToArm(const messages::JacoCartesianPose& jaco_cartesian_pose,
                              TrajectoryPoint& trajectory_point);

  // Assign arm position to cartesian_pose
  void assignCartesianPositionFromArm(const CartesianPosition& cartesian_pose_arm,
                                      messages::JacoCartesianPose& cartesian_pose);

  // Assign arm orientation to cartesian_pose (converting RPY Euler angles to quaternion)
  void assignOrientationFromArm(const CartesianPosition& cartesian_pose_arm,
                                messages::JacoCartesianPose& cartesian_pose);

    // Get observation of current position and orientation of end effector and transmit message
  void publishEndEffectorPose();

  // Get observation of current angular position for each joint and transmit message
  void publishJointPositions();

  // Get observation of current angular velocities for each joint and transmit message
  void publishJointVelocities();

  // Get observation of current finger position transmit message
  void publishFingerPositions();

  // Interface to Jaco API functions
  KinovaJacoAPI kinova_jaco_api_;

  // True if API has been successfully initialized
  bool successful_api_open_;

  // Acquire time for most recently received position command
  std::optional<int64_t> last_command_time_;
};

}  // namespace kinova_jaco
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::kinova_jaco::KinovaJaco);
