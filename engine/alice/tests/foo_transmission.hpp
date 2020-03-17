/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <atomic>
#include <functional>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"

namespace isaac {
namespace alice {

// @experimental
// Emits small messages
class FooTransmitter : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_TX(FooProto, foo)
  ISAAC_PARAM(int, offset, 42)
  ISAAC_PARAM(int, expected_tick_count)
  ISAAC_PARAM(int, expected_tick_count_tolerance, 0)
};

// @experimental
// Receives messages from BulkyTransmitter and checks there integrity
class FooReceiver : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_RX(FooProto, foo)
  ISAAC_PARAM(int, offset, 42)
  ISAAC_PARAM(int, count_tolerance, 1)
  ISAAC_PARAM(int, expected_tick_count)
  ISAAC_PARAM(int, expected_tick_count_tolerance, 0)
  std::function<void(FooReceiver*)> on_tick_callback;
  std::atomic<size_t> tick_count{0};

 private:
  bool is_periodic_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::FooTransmitter);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::FooReceiver);
