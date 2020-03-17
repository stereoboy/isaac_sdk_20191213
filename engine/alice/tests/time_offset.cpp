/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/components/TimeOffset.hpp"
#include "engine/alice/tests/messages.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class AcqtimeFoo : public Codelet {
 public:
  void start() override {
    set_tick_period("10Hz");
    tickPeriodically();
  }
  void tick() override {
    auto foo = tx_foo().initProto();
    foo.setCount(getTickCount());
    foo.setValue(42);
    foo.setText("x");
    tx_foo().publish(getTickCount() + get_acqoffset());
  }

  ISAAC_PROTO_TX(FooProto, foo);
  ISAAC_PARAM(int64_t, acqoffset, 0);
};

class TimeOffsetTester : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_foo1());
    synchronize(rx_foo1(), rx_foo2());
  }
  void tick() override {
    auto foo1 = rx_foo1().getProto();
    auto foo2 = rx_foo2().getProto();
    EXPECT_EQ(foo1.getCount(), foo2.getCount());
  }
  void stop() override {
    EXPECT_GT(getTickCount(), 0);
  }

  ISAAC_PROTO_RX(FooProto, foo1);
  ISAAC_PROTO_RX(FooProto, foo2);
};

TEST(Alice, ClockTicking) {
  Application app;

  Node* pub1_node = app.createMessageNode("pub1");
  auto* pub1 = pub1_node->addComponent<AcqtimeFoo>();
  pub1->async_set_acqoffset(100);

  Node* pub2_node = app.createMessageNode("pub2");
  auto* pub2 = pub2_node->addComponent<AcqtimeFoo>();
  pub2->async_set_acqoffset(-40);

  Node* offset1_node = app.createMessageNode("offset1");
  auto* offset1 = offset1_node->addComponent<TimeOffset>();
  offset1->async_set_input_channel("input");
  offset1->async_set_output_channel("output");
  offset1->async_set_acqtime_offset(-100);
  Connect(pub1->tx_foo(), offset1, "input");

  Node* offset2_node = app.createMessageNode("offset2");
  auto* offset2 = offset2_node->addComponent<TimeOffset>();
  offset2->async_set_input_channel("input");
  offset2->async_set_output_channel("output");
  offset2->async_set_acqtime_offset(40);
  Connect(pub2->tx_foo(), offset2, "input");

  Node* test_node = app.createMessageNode("test");
  auto* test = test_node->addComponent<TimeOffsetTester>();
  Connect(offset1, "output", test->rx_foo1());
  Connect(offset2, "output", test->rx_foo2());

  app.startWaitStop(0.50);
}


}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::AcqtimeFoo);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::TimeOffsetTester);
