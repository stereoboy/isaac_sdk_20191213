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

TEST(Alice, RecordReplay) {
  Uuid app_uuid;
  // Record
  {
    Application app;
    app_uuid = app.uuid();

    Node* publisher_node = app.createMessageNode("publisher");
    auto* publisher = publisher_node->addComponent<FooTransmitter>();
    publisher->async_set_tick_period("0.05s");
    publisher->async_set_expected_tick_count(11);
    publisher->async_set_expected_tick_count_tolerance(1);

    Node* recorder_node = app.createMessageNode("recorder");
    Recorder* recorder = recorder_node->addComponent<Recorder>();

    Connect(publisher->tx_foo(), recorder, "foo");

    app.startWaitStop(0.52);
  }

  // Replay
  {
    Application app;

    Node* replay_node = app.createMessageNode("replay");
    Replay* replay = replay_node->addComponent<Replay>();
    replay->async_set_cask_directory("/tmp/isaac/" + app_uuid.str());
    replay->async_set_loop(true);
    replay->async_set_use_recorded_message_time(false);

    Node* subscriber_node = app.createMessageNode("subscriber");
    auto* subscriber = subscriber_node->addComponent<FooReceiver>();
    subscriber->async_set_expected_tick_count(24);
    subscriber->async_set_expected_tick_count_tolerance(3);
    subscriber->async_set_count_tolerance(1000);  // disable

    Connect(replay, "foo", subscriber->rx_foo());

    app.startWaitStop(1.22);
  }
}

}  // namespace alice
}  // namespace isaac
