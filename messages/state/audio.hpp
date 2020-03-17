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

// Angle of the dominant sound source in radians
struct SourceAngleState : public state::State<double, 1> {
  // Counter-clockwise angle in radians
  ISAAC_STATE_VAR(0, angle);
};

// Average energy of an audio packet in decibels (dB)
struct AudioEnergyState : public state::State<double, 1> {
  // Energy in decibels (dB)
  ISAAC_STATE_VAR(0, energy);
};

}  // namespace messages
}  // namespace isaac
