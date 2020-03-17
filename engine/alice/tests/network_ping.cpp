/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
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
    outmsg.setValue(getTickCount());
    tx_out().publish();
  }
  ISAAC_PROTO_TX(IntProto, out)
};

class PingRx : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_in());
  }
  void tick() override {
    auto msg = rx_in().getProto();
    LOG_INFO("Received: %d", msg.getValue());
    if (isFirstTick()) {
      offset_ = msg.getValue() - 1;
    }
    EXPECT_EQ(offset_ + getTickCount(), msg.getValue());
  }
  ISAAC_PROTO_RX(IntProto, in)
 private:
  int offset_;
};

TEST(Alice, UdpPing) {
  constexpr int kPort = 42005;
  std::thread t1([] {
    Application app;
    // create publisher and subscriber objects
    Node* local_node = app.createMessageNode("local");
    auto* ping_rx = local_node->addComponent<PingRx>();
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<UdpSubscriber>();
    udp_sub->async_set_port(kPort);
    Connect(udp_sub, "something", ping_rx->rx_in());
    app.startWaitStop(2.50);  // run longer to give t2 a chance // TODO still unreliable though
    EXPECT_NEAR(ping_rx->getTickCount(), 30, 2);
  });
  Sleep(SecondsToNano(0.5));
  std::thread t2([] {
    Application app;
    // create publisher and subscriber objects
    Node* local_node = app.createMessageNode("local");
    auto* ping_tx = local_node->addComponent<PingTx>();
    ping_tx->async_set_tick_period("50ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<UdpPublisher>();
    udp_pub->async_set_host("127.0.0.1");
    udp_pub->async_set_port(kPort);
    Connect(ping_tx->tx_out(), udp_pub, "something");
    app.startWaitStop(1.50);
    Sleep(SecondsToNano(0.5));
    EXPECT_NEAR(ping_tx->getTickCount(), 30, 2);
  });
  t2.join();
  t1.join();
  // TODO this is quite unreliable as the two threads don't really have to run together...
}

TEST(Alice, TcpPing) {
  constexpr int kPort = 42006;
  std::thread t1([] {
    Application app;
    // create message transmitter
    Node* local_node = app.createMessageNode("local");
    auto* ping_tx = local_node->addComponent<PingTx>();
    ping_tx->async_set_tick_period("50ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<TcpPublisher>();
    udp_pub->async_set_port(kPort);
    Connect(ping_tx->tx_out(), udp_pub, "something");
    Sleep(SecondsToNano(0.5));
    app.startWaitStop(1.50);
    EXPECT_NEAR(ping_tx->getTickCount(), 30, 2);
  });
  Sleep(SecondsToNano(0.5));
  std::thread t2([] {
    Application app;
    // create message receiver
    Node* local_node = app.createMessageNode("local");
    auto* ping_rx = local_node->addComponent<PingRx>();
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<TcpSubscriber>();
    udp_sub->async_set_host("127.0.0.1");
    udp_sub->async_set_port(kPort);
    udp_sub->async_set_reconnect_interval(0.1);
    Connect(udp_sub, "something", ping_rx->rx_in());
    app.startWaitStop(2.00);  // run longer to give t2 a chance // TODO still unreliable though
    Sleep(SecondsToNano(0.5));
    EXPECT_NEAR(ping_rx->getTickCount(), 30, 2);
  });
  t2.join();
  t1.join();
  // TODO this is quite unreliable as the two threads don't really have to run together...
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingRx);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingTx);
