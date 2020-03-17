/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/gems/state/state.hpp"

namespace isaac {
namespace messages {

// Cartesian position of the end effector of the 7-dof JACO arm
struct JacoCartesianPose : public state::State<double, 7> {
  // End effector position
  ISAAC_STATE_VAR(0, px);
  ISAAC_STATE_VAR(1, py);
  ISAAC_STATE_VAR(2, pz);

  // End effector orientation
  ISAAC_STATE_VAR(3, qw);
  ISAAC_STATE_VAR(4, qx);
  ISAAC_STATE_VAR(5, qy);
  ISAAC_STATE_VAR(6, qz);

  // Sets the orientation from a quaternion
  void setOrientationFromQuaternion(const Quaterniond& quaternion) {
    this->qw() = quaternion.w();
    this->qx() = quaternion.x();
    this->qy() = quaternion.y();
    this->qz() = quaternion.z();
  }

  // Returns the quaternion representation of orientation
  Quaterniond quaternion() const {
    return Quaterniond(this->qw(), this->qx(), this->qy(), this->qz());
  }
};

// Joint position of the 7-dof JACO arm
struct JacoJointPosition : public state::State<double, 7> {
  ISAAC_STATE_VAR(0, actuator_1);
  ISAAC_STATE_VAR(1, actuator_2);
  ISAAC_STATE_VAR(2, actuator_3);
  ISAAC_STATE_VAR(3, actuator_4);
  ISAAC_STATE_VAR(4, actuator_5);
  ISAAC_STATE_VAR(5, actuator_6);
  ISAAC_STATE_VAR(6, actuator_7);
};

// Joint velocity of the 7-dof JACO arm
struct JacoJointVelocity : public state::State<double, 7> {
  ISAAC_STATE_VAR(0, actuator_1);
  ISAAC_STATE_VAR(1, actuator_2);
  ISAAC_STATE_VAR(2, actuator_3);
  ISAAC_STATE_VAR(3, actuator_4);
  ISAAC_STATE_VAR(4, actuator_5);
  ISAAC_STATE_VAR(5, actuator_6);
  ISAAC_STATE_VAR(6, actuator_7);
};


// State of the fingers
struct JacoFingerPosition : public state::State<double, 3> {
  ISAAC_STATE_VAR(0, finger_1);
  ISAAC_STATE_VAR(1, finger_2);
  ISAAC_STATE_VAR(2, finger_3);
};

}  // namespace messages
}  // namespace isaac
