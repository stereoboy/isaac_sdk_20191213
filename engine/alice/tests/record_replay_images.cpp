/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/tests/image_transmission.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, RecordReplayImages) {
  Uuid app_uuid;
  // Record
  {
    Application app;
    app_uuid = app.uuid();

    Node* transmitter_node = app.createMessageNode("transmitter");
    ImageTransmitter* transmitter = transmitter_node->addComponent<ImageTransmitter>();
    transmitter->async_set_tick_period("100ms");
    transmitter->async_set_rows(1000);
    transmitter->async_set_cols(1500);

    Node* recorder_node = app.createMessageNode("recorder");
    Recorder* recorder = recorder_node->addComponent<Recorder>();

    Connect(transmitter->tx_image(), recorder, "out");

    app.startWaitStop(4.50);
  }

  // Replay
  {
    Application app;

    Node* replay_node = app.createMessageNode("replay");
    Replay* replay = replay_node->addComponent<Replay>();
    replay->async_set_cask_directory("/tmp/isaac/" + app_uuid.str());

    Node* receiver_node = app.createMessageNode("receiver");
    ImageReceiver* receiver = receiver_node->addComponent<ImageReceiver>();
    receiver->async_set_rows(1000);
    receiver->async_set_cols(1500);

    Connect(replay, "out", receiver->rx_image());

    app.startWaitStop(4.70);
  }
}

}  // namespace alice
}  // namespace isaac
