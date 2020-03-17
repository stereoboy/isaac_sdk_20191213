/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <numeric>
#include <vector>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/foo_transmission.hpp"
#include "engine/core/assert.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

namespace {
constexpr int kNumPubSub = 30;
}  // namespace

// Checks if codelets are still ticking
class Monitor : public Codelet {
 public:
  void start() override {
    tickPeriodically();
    last_min_ = 0;
  }

  void tick() override {
    std::string summary;
    size_t min_count = std::numeric_limits<size_t>::max();
    size_t max_count = std::numeric_limits<size_t>::min();
    for (size_t i = 0; i < codelets_.size(); i++) {
      FooReceiver* codelet = codelets_[i];
      const size_t tick_count = codelet->tick_count;
      summary += std::to_string(tick_count) + ", ";
      min_count = std::min(min_count, tick_count);
      max_count = std::max(max_count, tick_count);
    }
    LOG_INFO("[%zu - %zu]: %s", min_count, max_count, summary.c_str());
    if (min_count == last_min_) {
      last_min_no_change_ ++;
      EXPECT_LT(last_min_no_change_, 3);
    } else {
      last_min_no_change_ = 0;
    }
    last_min_ = min_count;
  }

  void add(FooReceiver* node) {
    codelets_.push_back(node);
  }

 private:
  std::vector<FooReceiver*> codelets_;
  size_t last_min_;
  size_t last_min_no_change_;
};

TEST(Stresstest, PingStartStop) {
  Application app;

  // create monitor node
  Node* monitor_node = app.createNode("monitor");
  Monitor* monitor = monitor_node->addComponent<Monitor>();
  monitor->async_set_tick_period("100ms");

  // create publisher and subscriber nodes
  for (size_t i = 0; i < kNumPubSub; i++) {
    Node* pub_node = app.createMessageNode("pub_" + std::to_string(i));
    auto* pub = pub_node->addComponent<FooTransmitter>();
    pub->async_set_tick_period("1ms");
    Node* sub_node = app.createMessageNode("sub_" + std::to_string(i));
    auto* sub = sub_node->addComponent<FooReceiver>();
    sub->async_set_count_tolerance(1000000000);
    monitor->add(sub);
    Connect(pub->tx_foo(), sub->rx_foo());
  }

  // run for a while
  app.startWaitStop(10.0);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Monitor);
