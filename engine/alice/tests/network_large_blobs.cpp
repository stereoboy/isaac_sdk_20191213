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
#include "engine/alice/tests/bulky_transmission.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

namespace {

constexpr double kAliveTime = 2.0;
constexpr double kStartFudgeTime = 0.5;
constexpr double kStopFudgeTime = 2.5;

}  // namespace

TEST(Alice, UdpBulky) {
  // create receiver
  std::thread t2([] {
    Application app;
    // create publisher and subscriber objects
    Node* local_node = app.createMessageNode("local");
    auto* bulk_rx = local_node->addComponent<BulkyReceiver>();
    bulk_rx->async_set_chunk_size(1000);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<UdpSubscriber>();
    udp_sub->async_set_port(42001);
    Connect(udp_sub, "something", bulk_rx->rx_in());
    app.startWaitStop(kAliveTime + kStartFudgeTime + kStopFudgeTime);
    EXPECT_NEAR(bulk_rx->getTickCount(), 20, 4);
  });
  Sleep(SecondsToNano(kStartFudgeTime));
  // create transmitter
  std::thread t1([] {
    Application app;
    Node* local_node = app.createMessageNode("local");
    auto* bulk_tx = local_node->addComponent<BulkyTransmitter>();
    bulk_tx->async_set_chunk_size(1000);
    bulk_tx->async_set_tick_period("100ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<UdpPublisher>();
    udp_pub->async_set_host("localhost");
    udp_pub->async_set_port(42001);
    Connect(bulk_tx->tx_out(), udp_pub, "something");
    app.startWaitStop(kAliveTime);
    EXPECT_NEAR(bulk_tx->getTickCount(), 20, 4);
    Sleep(SecondsToNano(kStopFudgeTime));
  });
  t1.join();
  t2.join();
}

TEST(Alice, TcpBulky) {
  std::thread t1([] {
    Application app;
    // create message transmitter
    Node* local_node = app.createMessageNode("local_node");
    auto* bulk_tx = local_node->addComponent<BulkyTransmitter>();
    bulk_tx->async_set_chunk_size(100000);
    bulk_tx->async_set_tick_period("100ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<TcpPublisher>();
    udp_pub->async_set_port(42012);
    Connect(bulk_tx->tx_out(), udp_pub, "something");
    Sleep(SecondsToNano(0.5));
    app.startWaitStop(kAliveTime);
    EXPECT_NEAR(bulk_tx->getTickCount(), 20, 4);
  });
  Sleep(SecondsToNano(0.5));
  std::thread t2([] {
    Application app;
    // create message receiver
    Node* local_node = app.createMessageNode("local_node");
    auto* bulk_rx = local_node->addComponent<BulkyReceiver>();
    bulk_rx->async_set_chunk_size(100000);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<TcpSubscriber>();
    udp_sub->async_set_host("127.0.0.1");
    udp_sub->async_set_port(42012);
    udp_sub->async_set_reconnect_interval(0.1);
    Connect(udp_sub, "something", bulk_rx->rx_in());
    app.startWaitStop(kAliveTime + 0.5);  // run longer to give t2 a chance // TODO still unreliable though
    Sleep(SecondsToNano(0.5));
    EXPECT_NEAR(bulk_rx->getTickCount(), 20, 4);
  });
  t2.join();
  t1.join();
  // TODO this is quite unreliable as the two threads don't really have to run together...
}

}  // namespace alice
}  // namespace isaac
