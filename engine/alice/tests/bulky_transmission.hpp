/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <set>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

// @experimental
// Emits large messages
class BulkyTransmitter : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_TX(BulkyProto, out)
  ISAAC_PARAM(int, chunk_size, 100)
};

// @experimental
// Receives large messages from BulkyTransmitter and checks there integrity
class BulkyReceiver : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_RX(BulkyProto, in)
  ISAAC_PARAM(int, chunk_size, 100)
 private:
  std::set<Uuid> uuids_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::BulkyTransmitter);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::BulkyReceiver);
