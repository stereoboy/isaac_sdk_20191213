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

class TaskSeed : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    auto baro = tx_out().initProto();
    baro.setCounter(0);
    baro.setHop(0);
    tx_out().publish();
  }
  ISAAC_PROTO_TX(BarProto, out)
};

class Task : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_in());
  }
  void tick() override {
    ASSERT_TRUE(rx_in().available());
    auto bari = rx_in().getProto();
    auto baro = tx_out().initProto();
    baro.setCounter(bari.getCounter() + 1);
    baro.setHop(bari.getHop() + 1);
    tx_out().publish();
  }
  ISAAC_PROTO_RX(BarProto, in)
  ISAAC_PROTO_TX(BarProto, out)
};

TEST(Alice, MessagePassingCapnp) {
  Application app;
  Node* node1 = app.createMessageNode("node1");
  auto* task1 = node1->addComponent<Task>();
  Node* node2 = app.createMessageNode("node2");
  auto* task2 = node2->addComponent<Task>();
  Node* node3 = app.createMessageNode("node3");
  auto* task3 = node3->addComponent<Task>();
  Node* seed_node = app.createMessageNode("seed_node");
  auto* seed = seed_node->addComponent<TaskSeed>();
  seed->async_set_tick_period("50ms");
  Connect(task1->tx_out(), task2->rx_in());
  Connect(task2->tx_out(), task3->rx_in());
  Connect(task3->tx_out(), task1->rx_in());
  Connect(seed->tx_out(), task1->rx_in());
  Connect(seed->tx_out(), task2->rx_in());
  Connect(seed->tx_out(), task3->rx_in());

  app.startWaitStop(2.0);
  EXPECT_NEAR(seed->getTickCount(), 40, 5);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::TaskSeed);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Task);
