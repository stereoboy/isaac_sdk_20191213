/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyAlice : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    auto foo = tx_foo().initProto();
    foo.setCount(42);
    foo.setValue(1.17);
    foo.setText("red roses");
    tx_foo().publish();
  }
  ISAAC_PROTO_TX(FooProto, foo)
};

class MyBob : public Codelet {
 public:
  void start() override {
    auto* component = node()->app()->getNodeByName("alice")->getComponent<MyAlice>();
    Connect(component->tx_foo(), rx_foo());
    tickOnMessage(rx_foo());
  }
  void stop() override {
    EXPECT_NEAR(getTickCount(), 30, 3);
  }
  ISAAC_PROTO_RX(FooProto, foo)
};

// Tests that two nodes can be connected for message passing inside the start function of a Codelet.
// Normally they would be connected outside at application scope.
TEST(Alice, Config) {
  Application app;

  Node* alice_node = app.createMessageNode("alice");
  auto* alice = alice_node->addComponent<MyAlice>();
  alice->async_set_tick_period("10ms");

  Node* bob_node = app.createMessageNode("bob");
  bob_node->addComponent<MyBob>();

  app.startWaitStop(0.30);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyAlice);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyBob);
