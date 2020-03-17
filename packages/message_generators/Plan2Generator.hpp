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
#include "engine/core/math/types.hpp"
#include "messages/messages.hpp"

namespace isaac {
namespace message_generators {

// Plan2Generator creates plan which goes straight
class Plan2Generator : public alice::Codelet {
 public:
  void start() override;
  void tick() override;

  // The plan generated as specified via parameters
  ISAAC_PROTO_TX(Plan2Proto, plan);

  // Number of steps in the plan
  ISAAC_PARAM(int, count, 10);
  // The translation delta for every step in the plan
  ISAAC_PARAM(Vector2d, step, Vector2d(1.0, 0.0));
};

}  // namespace message_generators
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::message_generators::Plan2Generator);
