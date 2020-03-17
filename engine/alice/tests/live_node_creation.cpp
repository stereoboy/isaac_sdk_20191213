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
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/alice/tests/foo_transmission.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

std::vector<Node*> CreatePingPong(Application& app, int n) {
  const int offset = (219*n) % 31;
  LOG_INFO("Incursion: %d (offset: %d)", n, offset);
  // create publisher and subscriber nodes
  Node* pub_node = app.createMessageNode("pub_" + std::to_string(n));
  auto* pub = pub_node->addComponent<FooTransmitter>();
  pub->async_set_offset(offset);
  pub->async_set_tick_period("100ms");
  Node* sub_node = app.createMessageNode("sub_" + std::to_string(n));
  auto* sub = sub_node->addComponent<FooReceiver>();
  sub->async_set_count_tolerance(7);
  sub->async_set_offset(offset);
  Connect(pub->tx_foo(), sub->rx_foo());
  return {pub_node, sub_node};
}

TEST(MessagePassing, Test1) {
  Application app;
  std::vector<std::thread> threads;
  for (int i=0; i<10; i++) {
    threads.push_back(std::thread([i,&app] {
      auto nodes = CreatePingPong(app, i);
      Sleep(((i + 269)%13) * 50'000'000);
      app.backend()->node_backend()->startNodes(nodes);
      Sleep(((i + 71)%13) * 100'000'000);
      app.backend()->node_backend()->stopNodes(nodes);
    }));
  }
  app.start();
  for (auto& thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }
  app.stop();
}

}  // namespace alice
}  // namespace isaac
