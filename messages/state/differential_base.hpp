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

// State vector for a 2D base which can rotate around its origin, but only move in the direction
// of the X axis.
struct DifferentialBaseState : state::State<double, 7> {
  // Position of the base
  ISAAC_STATE_VAR(0, pos_x);
  ISAAC_STATE_VAR(1, pos_y);
  // Heading of the base
  ISAAC_STATE_VAR(2, heading);
  // Linear velocity of the base
  ISAAC_STATE_VAR(3, linear_speed);
  // Angular speed of the base
  ISAAC_STATE_VAR(4, angular_speed);
  // Linear acceleration of the base
  ISAAC_STATE_VAR(5, linear_acceleration);
  // Angular acceleration of the base
  ISAAC_STATE_VAR(6, angular_acceleration);
};

// Observation of the dynamics of the base
struct DifferentialBaseDynamics : state::State<double, 4> {
  // Linear velocity of the base
  ISAAC_STATE_VAR(0, linear_speed);
  // Angular speed of the base
  ISAAC_STATE_VAR(1, angular_speed);
  // Linear acceleration of the base
  ISAAC_STATE_VAR(2, linear_acceleration);
  // Angular acceleration of the base
  ISAAC_STATE_VAR(3, angular_acceleration);
};

// Controls used by a differential base
struct DifferentialBaseControl : state::State<double, 2> {
  // Linear velocity of the base
  ISAAC_STATE_VAR(0, linear_speed);
  // Angular speed of the base
  ISAAC_STATE_VAR(1, angular_speed);
};

}  // namespace messages
}  // namespace isaac
