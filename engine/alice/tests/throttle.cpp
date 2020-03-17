/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/tests/foo_transmission.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, ThrottleBySignal) {
  Application app;
  // Create data publisher
  Node* data_node = app.createMessageNode("data");
  auto* data = data_node->addComponent<FooTransmitter>();
  data->async_set_tick_period("10ms");
  data->async_set_expected_tick_count(60);
  data->async_set_expected_tick_count_tolerance(6);
  // Create signal publisher
  Node* signal_node = app.createMessageNode("signal");
  auto* signal = signal_node->addComponent<FooTransmitter>();
  signal->async_set_tick_period("100ms");
  signal->async_set_expected_tick_count(6);
  signal->async_set_expected_tick_count_tolerance(1);
  // Create throttle
  Node* throttle_node = app.createMessageNode("throttle");
  auto* throttle = throttle_node->addComponent<Throttle>();
  throttle->async_set_data_channel("data");
  throttle->async_set_output_channel("output");
  throttle->async_set_minimum_interval(0.0);
  throttle->async_set_use_signal_channel(true);
  throttle->async_set_signal_channel("signal");
  throttle->async_set_acqtime_tolerance(10'000'000);
  throttle_node->getComponent<MessageLedger>()->async_set_history(20);
  // Create subscriber
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<FooReceiver>();
  sub->async_set_expected_tick_count(6);
  sub->async_set_expected_tick_count_tolerance(1);
  sub->async_set_count_tolerance(1000);  // disable
  // Connections
  Connect(data->tx_foo(), throttle, "data");
  Connect(signal->tx_foo(), throttle, "signal");
  Connect(throttle, "output", sub->rx_foo());
  // run for a while
  app.startWaitStop(0.55);
}

TEST(Alice, ThrottleByTimer) {
  Application app;
  // Create data publisher
  Node* data_node = app.createMessageNode("data");
  auto* data = data_node->addComponent<FooTransmitter>();
  data->async_set_tick_period("10ms");
  data->async_set_expected_tick_count(60);
  data->async_set_expected_tick_count_tolerance(6);
  // Create throttle
  Node* throttle_node = app.createMessageNode("throttle");
  auto* throttle = throttle_node->addComponent<Throttle>();
  throttle->async_set_data_channel("data");
  throttle->async_set_output_channel("output");
  throttle->async_set_minimum_interval(0.1);
  throttle->async_set_use_signal_channel(false);
  // Create subscriber
  Node* sub_node = app.createMessageNode("sub");
  auto* sub = sub_node->addComponent<FooReceiver>();
  sub->async_set_expected_tick_count(6);
  sub->async_set_expected_tick_count_tolerance(1);
  sub->async_set_count_tolerance(1000);  // disable
  // Connections
  Connect(data->tx_foo(), throttle, "data");
  Connect(throttle, "output", sub->rx_foo());
  // run for a while
  app.startWaitStop(0.55);
}

}  // namespace alice
}  // namespace isaac
