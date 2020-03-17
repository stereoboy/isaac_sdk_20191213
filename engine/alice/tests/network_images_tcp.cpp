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
#include "engine/alice/tests/image_transmission.hpp"
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

TEST(Alice, TcpImages) {
  std::thread transmitter_thread([] {
    Application app;
    // create message transmitter
    Node* local_node = app.createMessageNode("local_node");
    auto* bulk_tx = local_node->addComponent<ImageTransmitter>();
    bulk_tx->async_set_rows(1000);
    bulk_tx->async_set_cols(1500);
    bulk_tx->async_set_tick_period("100ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<TcpPublisher>();
    udp_pub->async_set_port(42011);
    Connect(bulk_tx->tx_image(), udp_pub, "image");
    Sleep(SecondsToNano(0.5));
    app.startWaitStop(kAliveTime);
    EXPECT_NEAR(bulk_tx->getTickCount(), 20, 4);
  });
  Sleep(SecondsToNano(0.5));
  std::thread receiver_thread([] {
    Application app;
    // create message receiver
    Node* local_node = app.createMessageNode("local_node");
    auto* bulk_rx = local_node->addComponent<ImageReceiver>();
    bulk_rx->async_set_rows(1000);
    bulk_rx->async_set_cols(1500);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<TcpSubscriber>();
    udp_sub->async_set_host("127.0.0.1");
    udp_sub->async_set_port(42011);
    udp_sub->async_set_reconnect_interval(0.1);
    Connect(udp_sub, "image", bulk_rx->rx_image());
    app.startWaitStop(kAliveTime + 0.5);  // run longer to give t2 a chance // TODO still unreliable though
    Sleep(SecondsToNano(0.5));
    EXPECT_NEAR(bulk_rx->getTickCount(), 20, 4);
  });
  receiver_thread.join();
  transmitter_thread.join();
  // TODO this is quite unreliable as the two threads don't really have to run together...
}

}  // namespace alice
}  // namespace isaac
