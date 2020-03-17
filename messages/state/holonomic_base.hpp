/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

// State vector for a 2D holonomic base.
struct HolonomicBaseState : state::State<double, 8> {
  // Position of the base
  ISAAC_STATE_VAR(0, pos_x);
  ISAAC_STATE_VAR(1, pos_y);
  // Heading of the base
  ISAAC_STATE_VAR(2, heading);
  // Linear velocity of the base
  ISAAC_STATE_VAR(3, speed_x);
  ISAAC_STATE_VAR(4, speed_y);
  // Angular speed of the base
  ISAAC_STATE_VAR(5, angular_speed);
  // Linear acceleration of the base
  ISAAC_STATE_VAR(6, acceleration_x);
  ISAAC_STATE_VAR(7, acceleration_y);
};

// Observation of the dynamics of the holonomic base
struct HolonomicBaseDynamics : state::State<double, 5> {
  // Linear velocity of the base
  ISAAC_STATE_VAR(0, speed_x);
  ISAAC_STATE_VAR(1, speed_y);
  // Angular speed of the base
  ISAAC_STATE_VAR(2, angular_speed);
  // Linear acceleration of the base
  ISAAC_STATE_VAR(3, acceleration_x);
  ISAAC_STATE_VAR(4, acceleration_y);
};

// Controls used by a holonomic base
struct HolonomicBaseControls : state::State<double, 3> {
  // Linear velocity of the base
  ISAAC_STATE_VAR(0, speed_x);
  ISAAC_STATE_VAR(1, speed_y);
  // Angular speed of the base
  ISAAC_STATE_VAR(2, angular_speed);
};

}  // namespace messages
}  // namespace isaac
