/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <map>
#include <thread>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class PingTx : public Codelet {
 public:
  void start() override { tickPeriodically(); }
  void tick() override {
    auto outmsg = tx_out().initProto();
    outmsg.setValue(count);
    tx_out().publish();
    count++;
  }
  ISAAC_PROTO_TX(IntProto, out)
  int count = 0;
};

class PingRx : public Codelet {
 public:
  void start() override { tickOnMessage(rx_in()); }
  void tick() override {
    Sleep(1'000'000'000);
    count++;
  }
  ISAAC_PROTO_RX(IntProto, in)
  ISAAC_PARAM(int, sleep, 0);
  int count = 0;
};

// This test verifies that an event driven codelet with a deep event queue
// cannot stall the entire application.
TEST(Alice, EventStall) {
  Application app;
  // Create a ticking node and a node which ticks on message
  Node* local_node = app.createMessageNode("local");
  auto* ping_tx = local_node->addComponent<PingTx>();
  ping_tx->async_set_tick_period("50ms");
  auto* ping_rx = local_node->addComponent<PingRx>();

  // Set a very deep event queue depth greater than the number of worker threads.
  ping_rx->async_set_execution_queue_limit(1000);
  Connect(ping_tx->tx_out(), ping_rx->rx_in());
  app.startWaitStop(3);

  // The sender should tick at least this many times, but due to shut down delay
  // may tick more.
  EXPECT_GE(ping_tx->getTickCount(), 59);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingRx);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingTx);
