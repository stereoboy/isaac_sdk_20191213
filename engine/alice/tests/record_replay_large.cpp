/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/tests/bulky_transmission.hpp"
#include "engine/alice/tests/messages.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, RecordReplayLarge) {
  Uuid app_uuid;
  // Record
  {
    Application app;
    app_uuid = app.uuid();

    Node* publisher_node = app.createMessageNode("publisher");
    BulkyTransmitter* publisher = publisher_node->addComponent<BulkyTransmitter>();
    publisher->async_set_tick_period("100ms");
    publisher->async_set_chunk_size(400000);

    Node* recorder_node = app.createMessageNode("recorder");
    Recorder* recorder = recorder_node->addComponent<Recorder>();

    Connect(publisher->tx_out(), recorder, "out");

    app.startWaitStop(4.50);
  }

  // Replay
  {
    Application app;

    Node* replay_node = app.createMessageNode("replay");
    Replay* replay = replay_node->addComponent<Replay>();
    replay->async_set_cask_directory("/tmp/isaac/" + app_uuid.str());

    Node* subscriber_node = app.createMessageNode("subscriber");
    BulkyReceiver* subscriber = subscriber_node->addComponent<BulkyReceiver>();
    subscriber->async_set_chunk_size(400000);

    Connect(replay, "out", subscriber->rx_in());

    app.startWaitStop(4.70);
  }
}

}  // namespace alice
}  // namespace isaac
