/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyPublisher : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    Foo foo{42 + (int)getTickCount(), 3.1415};
    tx_foo().publish(foo);
  }
  void stop() override {
    EXPECT_NEAR(getTickCount(), 6, 1);
  }

  ISAAC_RAW_TX(Foo, foo)
};

class MySubscriber : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_foo());
  }
  void tick() override {
    ASSERT_TRUE(rx_foo().available());
    const Foo& foo = rx_foo().get();
    EXPECT_NEAR(foo.n, 42 + getTickCount(), 1);
    EXPECT_EQ(foo.x, 3.1415);
  }
  void stop() override {
    EXPECT_NEAR(getTickCount(), 6, 1);
  }

  ISAAC_RAW_RX(Foo, foo)
};

TEST(MessagePassing, Test1) {
  Application app;
  // create publisher and subscriber nodes
  Node* pub_node = app.createMessageNode("pub");
  auto* pub = pub_node->addComponent<MyPublisher>();
  pub->async_set_tick_period("100ms");
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<MySubscriber>();
  Connect(pub->tx_foo(), sub->rx_foo());
  // run for a while
  app.startWaitStop(0.55);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyPublisher);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MySubscriber);
