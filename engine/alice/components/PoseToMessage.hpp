/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>

#include "engine/alice/alice_codelet.hpp"
#include "messages/alice.capnp.h"

namespace isaac {
namespace alice {

// Reads desired pose from pose tree and publishes pose information as a message
class PoseToMessage : public Codelet {
 public:
  void start() override;
  void tick() override;

  // Outgoing pose message from pose tree
  ISAAC_PROTO_TX(PoseTreeEdgeProto, pose);

  // Name of the reference frame of the left side of the pose
  ISAAC_PARAM(std::string, lhs_frame);
  // Name of the reference frame of the right side of the pose
  ISAAC_PARAM(std::string, rhs_frame);
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PoseToMessage);
