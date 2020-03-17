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

TEST(Alice, UdpImages) {
  // create receiver
  std::thread receiver_thread([] {
    Application app;
    // create publisher and subscriber objects
    Node* local_node = app.createMessageNode("local");
    auto* bulk_rx = local_node->addComponent<ImageReceiver>();
    bulk_rx->async_set_rows(100);
    bulk_rx->async_set_cols(150);
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_sub = remote_node->addComponent<UdpSubscriber>();
    udp_sub->async_set_port(42000);
    Connect(udp_sub, "image", bulk_rx->rx_image());
    app.startWaitStop(kAliveTime + kStartFudgeTime + kStopFudgeTime);
    EXPECT_NEAR(bulk_rx->getTickCount(), 20, 4);
  });
  Sleep(SecondsToNano(kStartFudgeTime));
  // create transmitter
  std::thread transmitter_thread([] {
    Application app;
    Node* local_node = app.createMessageNode("local");
    auto* bulk_tx = local_node->addComponent<ImageTransmitter>();
    bulk_tx->async_set_rows(100);
    bulk_tx->async_set_cols(150);
    bulk_tx->async_set_tick_period("100ms");
    Node* remote_node = app.createMessageNode("remote");
    auto* udp_pub = remote_node->addComponent<UdpPublisher>();
    udp_pub->async_set_host("localhost");
    udp_pub->async_set_port(42000);
    Connect(bulk_tx->tx_image(), udp_pub, "image");
    app.startWaitStop(kAliveTime);
    EXPECT_NEAR(bulk_tx->getTickCount(), 20, 4);
    Sleep(SecondsToNano(kStopFudgeTime));
  });
  transmitter_thread.join();
  receiver_thread.join();
}

}  // namespace alice
}  // namespace isaac
