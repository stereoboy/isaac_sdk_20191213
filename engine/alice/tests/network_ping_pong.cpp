/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <atomic>
#include <thread>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class PingPong : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_ping());
    if (config().getFlag("trigger")) {
      auto outmsg = tx_ping().initProto();
      outmsg.setValue(2);
      tx_ping().publish();
      count = 1;
    } else {
      count = 0;
    }
  }
  void tick() override {
    auto inmsg = rx_ping().getProto();
    const int new_count = inmsg.getValue();
    EXPECT_EQ(count, new_count - 2);
    count = new_count;
    Sleep(10'000'000);
    auto outmsg = tx_ping().initProto();
    outmsg.setValue(count + 1);
    tx_ping().publish();
  }
  int count;

  ISAAC_PROTO_RX(IntProto, ping)
  ISAAC_PROTO_TX(IntProto, ping)
};

TEST(Alice, UdpPingPong) {
  std::thread t1([] {
    Application app;
    // create publisher and subscriber object
    Node* ping_node = app.createMessageNode("ping");
    auto* ping = ping_node->addComponent<PingPong>();
    ping->config().setFlag("trigger", true);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_rx = remote_node->addComponent<UdpSubscriber>();
    udp_rx->config().setInt("port", 42003);
    auto* udp_tx = remote_node->addComponent<UdpPublisher>();
    udp_tx->config().setString("host", "localhost");
    udp_tx->config().setInt("port", 42004);
    Connect(udp_rx, "ping", ping->rx_ping());
    Connect(ping->tx_ping(), udp_tx, "ping");
    app.startWaitStop(2.00);
    // TODO this is quite unreliable as the two threads don't really run together...
    EXPECT_NEAR(ping->count, 200, 10);
  });
  std::thread t2([] {
    Application app;
    // create publisher and subscriber objects
    Node* ping_node = app.createMessageNode("ping");
    auto* ping = ping_node->addComponent<PingPong>();
    ping->config().setFlag("trigger", false);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_rx = remote_node->addComponent<UdpSubscriber>();
    udp_rx->config().setInt("port", 42004);
    auto* udp_tx = remote_node->addComponent<UdpPublisher>();
    udp_tx->config().setString("host", "localhost");
    udp_tx->config().setInt("port", 42003);
    Connect(udp_rx, "ping", ping->rx_ping());
    Connect(ping->tx_ping(), udp_tx, "ping");
    app.startWaitStop(2.00);
    // TODO this is quite unreliable as the two threads don't really run together...
    EXPECT_NEAR(ping->count, 200, 10);
  });
  t2.join();
  t1.join();
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingPong);
