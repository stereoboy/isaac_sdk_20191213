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

// Publishes two streams with identical data and acqtime but with a time delay
class MyPublisher : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    Foo foo{42 + (int)getTickCount(), 3.1415};
    tx_foo1().publish(foo);
    Sleep(SecondsToNano(0.04));
    tx_foo2().publish(foo);
  }

  ISAAC_RAW_TX(Foo, foo1)
  ISAAC_RAW_TX(Foo, foo2)
};

// Receives two synchronized message streams and checks that synchronization works
class MySubscriber : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_foo1());
    tickOnMessage(rx_foo2());
    synchronize(rx_foo1(), rx_foo2());
  }
  void tick() override {
    ASSERT_TRUE(rx_foo1().available());
    ASSERT_TRUE(rx_foo2().available());
    const Foo& foo1 = rx_foo1().get();
    const Foo& foo2 = rx_foo2().get();
    // EXPECT_NEAR(foo1.n, 42 + getTickCount(), 1);
    EXPECT_EQ(foo2.n, foo1.n);
    // Sleep(SecondsToNano(0.07));
  }

  ISAAC_RAW_RX(Foo, foo1)
  ISAAC_RAW_RX(Foo, foo2)
};

// Receives multiple streams and checks that a codelet can tick on a message while synchronizing
// on some other channels.
class MySubscriber2 : public Codelet {
 public:
  void start() override {
    synchronize(rx_foo1(), rx_foo2());
    tickOnMessage(rx_foo3());
    count_ = 0;
  }
  void tick() override {
    ASSERT_TRUE(rx_foo1().available() == rx_foo2().available());
    ASSERT_TRUE(rx_foo3().available());
    count_ ++;
  }
  void stop() override {
    EXPECT_NEAR(count_, get_expected_foo3_count(), get_count_tolerance());
  }
  ISAAC_PARAM(int, expected_foo3_count)
  ISAAC_PARAM(int, count_tolerance)
  ISAAC_RAW_RX(Foo, foo1)
  ISAAC_RAW_RX(Foo, foo2)
  ISAAC_RAW_RX(Foo, foo3)

 private:
  int count_;
};

// Publishes two streams with identical data and acqtime but with a time delay
class PublishWithAcqtimeCount : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    Foo foo{42 + (int)getTickCount(), 3.1415};
    tx_foo().publish(foo, getTickCount());
    Sleep(SecondsToNano(0.04));
  }
  ISAAC_RAW_TX(Foo, foo);
};

// Receives and synchronizes many message streams.
class SynchronizeMany : public Codelet {
 public:
  void start() override {
    synchronize(rx_foo1(), rx_foo2());
    synchronize(rx_foo1(), rx_foo3());
    synchronize(rx_foo1(), rx_foo4());
    synchronize(rx_foo1(), rx_foo5());
    synchronize(rx_foo1(), rx_foo6());
    synchronize(rx_foo1(), rx_foo7());
    synchronize(rx_foo1(), rx_foo8());
    tickOnMessage(rx_foo1());
  }
  void tick() override {
    ASSERT_TRUE(rx_foo1().available());
    ASSERT_TRUE(rx_foo2().available());
    ASSERT_TRUE(rx_foo3().available());
    ASSERT_TRUE(rx_foo4().available());
    ASSERT_TRUE(rx_foo5().available());
    ASSERT_TRUE(rx_foo6().available());
    ASSERT_TRUE(rx_foo7().available());
    ASSERT_TRUE(rx_foo8().available());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo2().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo3().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo4().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo5().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo6().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo7().acqtime());
    ASSERT_EQ(rx_foo1().acqtime(), rx_foo8().acqtime());
  }
  ISAAC_RAW_RX(Foo, foo1);
  ISAAC_RAW_RX(Foo, foo2);
  ISAAC_RAW_RX(Foo, foo3);
  ISAAC_RAW_RX(Foo, foo4);
  ISAAC_RAW_RX(Foo, foo5);
  ISAAC_RAW_RX(Foo, foo6);
  ISAAC_RAW_RX(Foo, foo7);
  ISAAC_RAW_RX(Foo, foo8);
};

TEST(MessagePassing, Test1) {
  Application app;
  // create publisher and subscriber nodes
  Node* pub_node = app.createMessageNode("pub");
  auto* pub = pub_node->addComponent<MyPublisher>();
  pub->async_set_tick_period("50ms");
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<MySubscriber>();
  Connect(pub->tx_foo1(), sub->rx_foo1());
  Connect(pub->tx_foo2(), sub->rx_foo2());
  // run for a while
  app.startWaitStop(1.00);
}

TEST(MessagePassing, Test2) {
  Application app;
  // create publisher and subscriber nodes
  Node* pub_node = app.createMessageNode("pub");
  auto* pub = pub_node->addComponent<MyPublisher>();
  pub->async_set_tick_period("50ms");
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<MySubscriber2>();
  sub->async_set_expected_foo3_count(20);
  sub->async_set_count_tolerance(2);
  Connect(pub->tx_foo1(), sub->rx_foo3());
  // run for a while
  app.startWaitStop(1.00);
}

TEST(MessagePassing, SynchronizeMany) {
  constexpr int kPubTickPeriod = 50;
  constexpr double kTotalTime = 2.0;
  constexpr int kTickCountTolerance = 5;
  Application app;
  // create a subscriber which will receive many messages
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<SynchronizeMany>();
  std::vector<const RawRx<Foo>*> sub_rx{{
    &(sub->rx_foo1()),
    &(sub->rx_foo2()),
    &(sub->rx_foo3()),
    &(sub->rx_foo4()),
    &(sub->rx_foo5()),
    &(sub->rx_foo6()),
    &(sub->rx_foo7()),
    &(sub->rx_foo8())
  }};
  // create many publisher nodes
  std::vector<PublishWithAcqtimeCount*> pubs;
  for (size_t i = 0; i < sub_rx.size(); i++) {
    Node* pub_node = app.createMessageNode("pub_" + std::to_string(i + 1));
    auto* pub = pub_node->addComponent<PublishWithAcqtimeCount>();
    pubs.push_back(pub);
    pub->async_set_tick_period(std::to_string(kPubTickPeriod) + "ms");
    Connect(pub->tx_foo(), *(sub_rx[i]));
  }
  ASSERT_EQ(pubs.size(), sub_rx.size());
  // run for a while
  app.startWaitStop(kTotalTime);
  // Check that all codelets ticked as expected
  constexpr int kExpectedTickCount =
      static_cast<double>(kTotalTime * 1000.0 / static_cast<double>(kPubTickPeriod));
  static_assert(kExpectedTickCount >= 2*kTickCountTolerance, "Parameters should be set such that "
                "expected tick count is greater than twice the tolerance.");
  for (size_t i = 0; i < sub_rx.size(); i++) {
    EXPECT_NEAR(pubs[i]->getTickCount(), kExpectedTickCount, kTickCountTolerance);
  }
  EXPECT_NEAR(sub->getTickCount(), kExpectedTickCount, kTickCountTolerance);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyPublisher);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MySubscriber);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MySubscriber2);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PublishWithAcqtimeCount);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::SynchronizeMany);
