/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once
#include <memory>

#include "capnp/compat/json.h"

#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/hooks/message_hook.hpp"
#include "engine/alice/message.hpp"
#include "engine/gems/math/exponential_moving_average.hpp"
#include "messages/messages.hpp"

namespace isaac {
namespace alice {

// Converts JSON messages into proto messages.
//
// JSON messages must be published on the channel "json". Note that the input channel does not
// appear in the normal list of channels due to how this codelet works internally.
//
// Type ID must be set correctly otherwise conversion will fail.
class JsonToProto : public Codelet {
 public:
  void start() override;
  void tick() override;

  // Publish proto messages in registered proto definition as specified by incoming json message
  // proto id.
  ISAAC_PROTO_TX(MessageHeaderProto, proto);

 private:
  // Average samples per second for visualization
  std::unique_ptr<::capnp::JsonCodec> json_codec_;
  MessageLedger* ledger_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::JsonToProto)
