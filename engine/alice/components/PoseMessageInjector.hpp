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
#include "messages/alice.capnp.h"

namespace isaac {
namespace alice {

// Receives pose information via messages and injects them into the pose tree.
class PoseMessageInjector : public Codelet {
 public:
  void start() override;
  void tick() override;

  // Incoming pose messages to inject into the pose tree
  ISAAC_PROTO_RX(PoseTreeEdgeProto, pose);
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PoseMessageInjector);
