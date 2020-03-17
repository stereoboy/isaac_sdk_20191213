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

// State of a pan/tilt unit which can rotate around two axes
struct PanTiltState : public state::State<double, 4> {
  ISAAC_STATE_VAR(0, pan);
  ISAAC_STATE_VAR(1, tilt);
  ISAAC_STATE_VAR(2, pan_speed);
  ISAAC_STATE_VAR(3, tilt_speed);
};

}  // namespace messages
}  // namespace isaac
