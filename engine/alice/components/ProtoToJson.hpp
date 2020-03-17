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

// Converts Proto messages into Json messages.
// Accepts all registered proto messages to channel "proto".
// Require valid type id from messages.
class ProtoToJson : public Codelet {
 public:
  void start() override;
  void tick() override;

  // Publishes converted Json message
  ISAAC_RAW_TX(nlohmann::json, json);

 private:
  // Average samples per second for visualization
  std::unique_ptr<::capnp::JsonCodec> json_codec_;
  MessageLedger* ledger_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ProtoToJson)
