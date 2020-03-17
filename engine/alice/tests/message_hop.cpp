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
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class Task1 : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    auto bar = tx_bar().initProto();
    bar.setCounter(getTickCount());
    bar.setHop(1);
    tx_bar().publish();
  }

  ISAAC_PROTO_TX(BarProto, bar)
};

class Task2 : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_in());
  }
  void tick() override {
    ASSERT_TRUE(rx_in().available());
    auto bari = rx_in().getProto();
    EXPECT_EQ(bari.getHop(), 1);
    auto baro = tx_out().initProto();
    baro.setCounter(bari.getCounter());
    baro.setHop(2);
    tx_out().publish();
  }

  ISAAC_PROTO_RX(BarProto, in)
  ISAAC_PROTO_TX(BarProto, out)
};

class Task3 : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_final());
  }
  void tick() override {
    ASSERT_TRUE(rx_final().available());
    auto bar = rx_final().getProto();
    EXPECT_NEAR(bar.getCounter(), getTickCount(), 1);
    EXPECT_EQ(bar.getHop(), 2);
  }

  ISAAC_PROTO_RX(BarProto, final)
};

TEST(Alice, MessagePassingCapnp) {
  Application app;
  Node* node1 = app.createMessageNode("node1");
  auto* task1 = node1->addComponent<Task1>();
  task1->async_set_tick_period("40ms");
  Node* node2 = app.createMessageNode("node2");
  auto* task2 = node2->addComponent<Task2>();
  Node* node3 = app.createMessageNode("node3");
  auto* task3 = node3->addComponent<Task3>();

  Connect(task1->tx_bar(), task2->rx_in());
  Connect(task2->tx_out(), task3->rx_final());

  app.startWaitStop(2.0);
  EXPECT_NEAR(task3->getTickCount(), 50, 5);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Task1);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Task2);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Task3);
