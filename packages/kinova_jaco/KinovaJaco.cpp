/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "KinovaJaco.hpp"

#include "KinovaTypes.h"
#include "engine/alice/alice.hpp"
#include "engine/alice/message.hpp"
#include "engine/gems/state/io.hpp"
#include "engine/gems/state/state.hpp"

namespace isaac {
namespace kinova_jaco {

void KinovaJaco::start() {
  // Load Jaco SDK and initialize USB connection with the arm
  successful_api_open_ = kinova_jaco_api_.open(get_kinova_jaco_sdk_path());

  if (successful_api_open_) {
    resetArm();
  }

  tickPeriodically();
}

void KinovaJaco::tick() {
  if (successful_api_open_) {
    // Send a cartesian position command if there is a new cartesian pose command message
    if (get_control_mode() == kCartesianPose && rx_cartesian_pose_command().available()) {
      parseCartesianPoseCommand();
    }

    // Send a joint velocity command if there is a new joint velocity command available
    if (get_control_mode() == kJointVelocity && rx_joint_velocity_command().available()) {
      parseJointVelocityCommand();
    }

    // Publish current state of the arm (cartesian pose of end effector, joint angles/velocities
    // and finger position.
    publishEndEffectorPose();
    publishJointPositions();
    publishJointVelocities();
    publishFingerPositions();
  }
}

void KinovaJaco::stop() {
  // Close connection with Jaco arm
  if (successful_api_open_) {
    // Remove all remaining points in the FIFO
    kinova_jaco_api_.eraseAllTrajectories();

    kinova_jaco_api_.close();
  }
}

void KinovaJaco::resetArm() {
  // Move finger to fully outstretched, ready to grasp
  kinova_jaco_api_.initFingers();

  kinova_jaco_api_.moveHome();
}

void KinovaJaco::parseCartesianPoseCommand() {
  const int64_t time = rx_cartesian_pose_command().acqtime();

  // If message is the first received message (last_command_time_ not initialized) or more recent
  // than last command, parse pose and send to arm.
  if (!last_command_time_ || time >= last_command_time_.value()) {
    last_command_time_ = time;

    // Read the commanded pose message and store as messages::JacoCartesianPose
    messages::JacoCartesianPose cartesian_pose_command;
    FromProto(rx_cartesian_pose_command().getProto(), rx_cartesian_pose_command().buffers(),
              cartesian_pose_command);

    // Assign commanded pose to arm (handling orientation conversions)
    TrajectoryPoint trajectory_point;
    initializeCartesianTrajectoryPoint(trajectory_point);
    assignCartesianPositionToArm(cartesian_pose_command, trajectory_point);
    assignOrientationToArm(cartesian_pose_command, trajectory_point);

    // Send trajectory point to arm
    kinova_jaco_api_.sendBasicTrajectory(trajectory_point);
  }
}

void KinovaJaco::parseJointVelocityCommand() {
  const int64_t time = rx_joint_velocity_command().acqtime();

  // If message is the first received message (last_command_time_ not initialized) or more recent
  // than last command, parse pose and send to arm.
  if (!last_command_time_ || time >= last_command_time_.value()) {
    last_command_time_ = time;

    // Read the commanded joint velocity message and store as messages::JacoJointVelocity
    messages::JacoJointVelocity joint_velocity_command;
    FromProto(rx_joint_velocity_command().getProto(), rx_joint_velocity_command().buffers(),
              joint_velocity_command);

    // Assign commanded joint velocities to arm
    TrajectoryPoint trajectory_point;
    initializeJointVelocityTrajectoryPoint(trajectory_point);

    auto& actuators = trajectory_point.Position.Actuators;
    actuators.Actuator1 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_1()));
    actuators.Actuator2 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_2()));
    actuators.Actuator3 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_3()));
    actuators.Actuator4 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_4()));
    actuators.Actuator5 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_5()));
    actuators.Actuator6 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_6()));
    actuators.Actuator7 = static_cast<float>(RadToDeg(joint_velocity_command.actuator_7()));

    // Send trajectory point to arm
    kinova_jaco_api_.sendBasicTrajectory(trajectory_point);
  }
}

void KinovaJaco::initializeTrajectoryPoint(TrajectoryPoint& trajectory_point) {
  trajectory_point.InitStruct();
  trajectory_point.LimitationsActive = 0;
  trajectory_point.Position.HandMode = HAND_NOMOVEMENT;
}

void KinovaJaco::initializeCartesianTrajectoryPoint(TrajectoryPoint& trajectory_point) {
  initializeTrajectoryPoint(trajectory_point);
  trajectory_point.Position.Type = CARTESIAN_POSITION;
}

void KinovaJaco::initializeJointVelocityTrajectoryPoint(TrajectoryPoint& trajectory_point) {
  initializeTrajectoryPoint(trajectory_point);
  trajectory_point.Position.Type = ANGULAR_VELOCITY;
}

void KinovaJaco::assignCartesianPositionToArm(
    const messages::JacoCartesianPose& jaco_cartesian_pose, TrajectoryPoint& trajectory_point) {
  CartesianInfo& cartesian_info = trajectory_point.Position.CartesianPosition;
  cartesian_info.X = static_cast<float>(jaco_cartesian_pose.px());
  cartesian_info.Y = static_cast<float>(jaco_cartesian_pose.py());
  cartesian_info.Z = static_cast<float>(jaco_cartesian_pose.pz());
}

void KinovaJaco::assignOrientationToArm(const messages::JacoCartesianPose& jaco_cartesian_pose,
                                        TrajectoryPoint& trajectory_point) {
  // Convert from quaternion to euler angles (roll/X, pitch/Y, yaw/Z)
  auto orientation_SO3 = SO3d::FromQuaternion(jaco_cartesian_pose.quaternion());
  auto orientation_euler_angles = orientation_SO3.eulerAnglesRPY();

  enum EulerAngles { kEulerX = 0, kEulerY = 1, kEulerZ = 2 };

  // Convert from radians to degrees to send to arm
  CartesianInfo& cartesian_info = trajectory_point.Position.CartesianPosition;
  cartesian_info.ThetaX = static_cast<float>(orientation_euler_angles[kEulerX]);
  cartesian_info.ThetaY = static_cast<float>(orientation_euler_angles[kEulerY]);
  cartesian_info.ThetaZ = static_cast<float>(orientation_euler_angles[kEulerZ]);
}

void KinovaJaco::publishEndEffectorPose() {
  // Read cartesian pose from arm
  CartesianPosition cartesian_pose_observation;
  kinova_jaco_api_.getCartesianPosition(cartesian_pose_observation);

  // Define messages::JacoCartesianPose to send as message
  messages::JacoCartesianPose cartesian_pose;
  assignCartesianPositionFromArm(cartesian_pose_observation, cartesian_pose);
  assignOrientationFromArm(cartesian_pose_observation, cartesian_pose);

  // Serialize and publish cartesian pose
  ToProto(cartesian_pose, tx_cartesian_pose().initProto(), tx_cartesian_pose().buffers());
  tx_cartesian_pose().publish();
}

void KinovaJaco::assignCartesianPositionFromArm(const CartesianPosition& cartesian_pose_arm,
                                                messages::JacoCartesianPose& cartesian_pose) {
  cartesian_pose.px() = cartesian_pose_arm.Coordinates.X;
  cartesian_pose.py() = cartesian_pose_arm.Coordinates.Y;
  cartesian_pose.pz() = cartesian_pose_arm.Coordinates.Z;
}

void KinovaJaco::assignOrientationFromArm(const CartesianPosition& cartesian_pose_arm,
                                          messages::JacoCartesianPose& cartesian_pose) {
  // Convert observed Euler angles from degrees to radians
  const double theta_x = cartesian_pose_arm.Coordinates.ThetaX;
  const double theta_y = cartesian_pose_arm.Coordinates.ThetaY;
  const double theta_z = cartesian_pose_arm.Coordinates.ThetaZ;

  // Convert from Euler angles (radians) to quaternion
  auto orientation_SO3 = SO3d::FromEulerAnglesRPY(theta_x, theta_y, theta_z);

  // Store observed orientation
  cartesian_pose.setOrientationFromQuaternion(orientation_SO3.quaternion());
}

void KinovaJaco::publishJointPositions() {
  // Read joint position from arm
  AngularPosition joint_position_observation;
  kinova_jaco_api_.getAngularPosition(joint_position_observation);

  // Define messages::JacoJointPosition to send as message
  messages::JacoJointPosition joint_position;

  // Store observed joint position (converting from degrees to radians)
  joint_position.actuator_1() = DegToRad(joint_position_observation.Actuators.Actuator1);
  joint_position.actuator_2() = DegToRad(joint_position_observation.Actuators.Actuator2);
  joint_position.actuator_3() = DegToRad(joint_position_observation.Actuators.Actuator3);
  joint_position.actuator_4() = DegToRad(joint_position_observation.Actuators.Actuator4);
  joint_position.actuator_5() = DegToRad(joint_position_observation.Actuators.Actuator5);
  joint_position.actuator_6() = DegToRad(joint_position_observation.Actuators.Actuator6);
  joint_position.actuator_7() = DegToRad(joint_position_observation.Actuators.Actuator7);

  // Serialize and publish joint position
  ToProto(joint_position, tx_joint_position().initProto(), tx_joint_position().buffers());
  tx_joint_position().publish();
}

void KinovaJaco::publishJointVelocities() {
  // Read joint velocity from arm
  AngularPosition joint_velocity_observation;
  kinova_jaco_api_.getAngularVelocity(joint_velocity_observation);

  // Define messages::JacoJointVelocity to send as message
  messages::JacoJointVelocity joint_velocity;

  // Store observed joint velocity (converting from degrees to radians)
  joint_velocity.actuator_1() = DegToRad(joint_velocity_observation.Actuators.Actuator1);
  joint_velocity.actuator_2() = DegToRad(joint_velocity_observation.Actuators.Actuator2);
  joint_velocity.actuator_3() = DegToRad(joint_velocity_observation.Actuators.Actuator3);
  joint_velocity.actuator_4() = DegToRad(joint_velocity_observation.Actuators.Actuator4);
  joint_velocity.actuator_5() = DegToRad(joint_velocity_observation.Actuators.Actuator5);
  joint_velocity.actuator_6() = DegToRad(joint_velocity_observation.Actuators.Actuator6);
  joint_velocity.actuator_7() = DegToRad(joint_velocity_observation.Actuators.Actuator7);

  // Serialize and publish joint velocity
  ToProto(joint_velocity, tx_joint_velocity().initProto(), tx_joint_velocity().buffers());
  tx_joint_velocity().publish();
}

void KinovaJaco::publishFingerPositions() {
  // Read cartesian position from arm (no method to read only fingers)
  CartesianPosition cartesian_position_observation;
  kinova_jaco_api_.getCartesianPosition(cartesian_position_observation);

  // Extract finger position from cartesian position
  FingersPosition finger_position_observation = cartesian_position_observation.Fingers;

  // Define messages::JacoFingerPosition to send as message
  messages::JacoFingerPosition finger_position;

  // Store observed finger position
  finger_position.finger_1() = finger_position_observation.Finger1;
  finger_position.finger_2() = finger_position_observation.Finger2;
  finger_position.finger_3() = finger_position_observation.Finger3;

  // Serialize and publish finger position
  ToProto(finger_position, tx_finger_position().initProto(), tx_finger_position().buffers());
  tx_finger_position().publish();
}

}  // namespace kinova_jaco
}  // namespace isaac
