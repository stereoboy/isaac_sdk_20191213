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
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class PingTx : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    auto outmsg = tx_out().initProto();
    outmsg.setValue(get_round());
    tx_out().publish();
  }
  ISAAC_PROTO_TX(IntProto, out)
  ISAAC_PARAM(int, round)
};

class PingRx : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_in());
  }
  void tick() override {
    counts[rx_in().getProto().getValue()]++;
  }
  ISAAC_PROTO_RX(IntProto, in)
  std::map<int, int> counts;
};

void TcpPublisherDeathImpl(int num_iterations, double duration, int port) {
  const double tick_period = 0.05;
  const double total_duration = static_cast<double>(num_iterations + 2) * duration;
  const int tick_count = std::floor(duration / tick_period);
  auto pub = [=](int round) {
    Application app;
    // create message transmitter
    Node* local_node = app.createMessageNode("local");
    auto* ping_tx = local_node->addComponent<PingTx>();
    ping_tx->async_set_round(round);
    ping_tx->async_set_tick_period(std::to_string(tick_period) + "s");
    Node* remote_node = app.createMessageNode("remote");
    auto* publisher = remote_node->addComponent<TcpPublisher>();
    publisher->async_set_port(port);
    Connect(ping_tx->tx_out(), publisher, "something");
    app.startWaitStop(duration);
    EXPECT_NEAR(ping_tx->getTickCount(), tick_count, 3);
  };
  auto sub = [=] {
    Application app;
    // create message receiver
    Node* local_node = app.createMessageNode("local");
    auto* ping_rx = local_node->addComponent<PingRx>();
    Node* remote_node = app.createMessageNode("remote");
    auto* subscriber = remote_node->addComponent<TcpSubscriber>();
    subscriber->async_set_host("localhost");
    subscriber->async_set_port(port);
    subscriber->async_set_reconnect_interval(0.1);
    Connect(subscriber, "something", ping_rx->rx_in());
    app.startWaitStop(total_duration);
    for (int i = 0; i < num_iterations; i++) {
      EXPECT_NEAR(ping_rx->counts[i], tick_count, 2);
    }
  };
  // start publisher, stop publisher, restart publisher to check if subscriber reconnects
  std::thread sub_thread(sub);
  for (int i = 0; i < num_iterations; i++) {
    std::thread pub_thread([=] { pub(i); });
    pub_thread.join();
  }
  sub_thread.join();
}

TEST(Alice, TcpPublisherDeath) {
  TcpPublisherDeathImpl(2, 1.0, 42007);
  TcpPublisherDeathImpl(4, 0.5, 42007);
}

void TcpReceiverDeathImpl(int num_iterations, double duration, int port) {
  const double tick_period = 0.05;
  const double total_duration = static_cast<double>(num_iterations + 2) * duration;
  const int total_tick_count = std::floor(total_duration / tick_period);
  const int tick_count = std::floor(duration / tick_period);
  auto pub = [=]() {
    Application app;
    // create message transmitter
    Node* local_node = app.createMessageNode("local");
    auto* ping_tx = local_node->addComponent<PingTx>();
    ping_tx->async_set_round(0);
    ping_tx->async_set_tick_period(std::to_string(tick_period) + "s");
    Node* remote_node = app.createMessageNode("remote");
    auto* publisher = remote_node->addComponent<TcpPublisher>();
    publisher->async_set_port(port);
    Connect(ping_tx->tx_out(), publisher, "something");
    app.startWaitStop(total_duration);
    EXPECT_NEAR(ping_tx->getTickCount(), total_tick_count, 2);
  };
  auto sub = [=] {
    Application app;
    // create message receiver
    Node* local_node = app.createMessageNode("local");
    auto* ping_rx = local_node->addComponent<PingRx>();
    Node* remote_node = app.createMessageNode("remote");
    auto* subscriber = remote_node->addComponent<TcpSubscriber>();
    subscriber->async_set_host("localhost");
    subscriber->async_set_port(port);
    subscriber->async_set_reconnect_interval(0.1);
    Connect(subscriber, "something", ping_rx->rx_in());
    app.startWaitStop(duration);
    EXPECT_NEAR(ping_rx->counts[0], tick_count, 2);
  };
  // start publisher, stop publisher, restart publisher to check if subscriber reconnects
  std::thread pub_thread(pub);
  for (int i = 0; i < num_iterations; i++) {
    std::thread sub_thread(sub);
    sub_thread.join();
  }
  pub_thread.join();
}

TEST(Alice, TcpReceiverDeath) {
  TcpReceiverDeathImpl(2, 1.0, 42008);
  TcpReceiverDeathImpl(4, 0.5, 42008);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingRx);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingTx);
