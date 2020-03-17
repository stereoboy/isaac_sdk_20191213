/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <atomic>

#include "engine/alice/alice.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/alice/tests/foo_transmission.hpp"
#include "engine/alice/tests/messages.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

std::atomic<int> incursions(0);

void CreatePingPong(Application& app) {
  const int n = incursions.fetch_add(1);
  const int offset = (197*n) % 137;
  LOG_INFO("Incursion: %d (offset: %d)", n, offset);
  // create publisher and subscriber nodes
  Node* pub_node = app.createMessageNode("pub_" + std::to_string(n));
  auto* pub = pub_node->addComponent<FooTransmitter>();
  pub->async_set_tick_period("0.10");
  pub->async_set_offset(offset);
  Node* sub_node = app.createMessageNode("sub_" + std::to_string(n));
  auto* sub = sub_node->addComponent<FooReceiver>();
  sub->async_set_offset(offset);
  sub->on_tick_callback = [&](FooReceiver* receiver) {
    if (!receiver->isFirstTick() && receiver->getTickCount() % 5 == 1) {
      CreatePingPong(app);
    }
  };
  Connect(pub->tx_foo(), sub->rx_foo());
  app.backend()->node_backend()->startNodes({pub_node, sub_node});
}

TEST(MessagePassing, Test1) {
  Application app;
  CreatePingPong(app);
  app.startWaitStop(2.30);
  EXPECT_NEAR(incursions, 15, 5);
}

}  // namespace alice
}  // namespace isaac
