/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <atomic>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

std::atomic<int> count(0);

class PingPong : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_value());
    if (count.load() == 0) {
      auto outmsg = tx_inc().initProto();
      count = 1;
      outmsg.setValue(1);
      tx_inc().publish();
    }
  }
  void tick() override {
    auto inmsg = rx_value().getProto();
    const int value = inmsg.getValue() + 1;
    count = value;
    Sleep(10'000'000);
    auto outmsg = tx_inc().initProto();
    outmsg.setValue(value);
    tx_inc().publish();
  }
 private:
  ISAAC_PROTO_RX(IntProto, value);
  ISAAC_PROTO_TX(IntProto, inc);
};

TEST(Alice, PingPong) {
  Application app;
  // create publisher and subscriber nodes
  Node* ping_node = app.createMessageNode("ping");
  auto* ping = ping_node->addComponent<PingPong>();
  Node* pong_node = app.createMessageNode("pong");
  auto* pong = pong_node->addComponent<PingPong>();
  // connect them together
  Connect(pong->tx_inc(), ping->rx_value());
  Connect(ping->tx_inc(), pong->rx_value());
  // run for a while
  app.startWaitStop(0.19);
  EXPECT_NEAR(count.load(), 20, 2);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingPong);
