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

// A publisher for cartesian pose transmitted by the Jaco arm.
// Prints to console the cartesian pose of the end effector when messages are received.
class KinovaJacoSamplePublisher : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override {}

  // Current position and orientation of end effector
  ISAAC_PROTO_RX(StateProto, cartesian_pose);
};

}  // namespace kinova_jaco
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::kinova_jaco::KinovaJacoSamplePublisher);
