/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <limits>

#include "engine/alice/alice.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class CodeletTickPeriodMonitor : public alice::Codelet {
 public:
  void start() {
    tickPeriodically();
    target_interval_ = SecondsToNano(getTickPeriodAsSeconds().value());
    tolerance_ = get_tolerance_dt();
    last_tick_ts_ = 0;
    min_interval_ = std::numeric_limits<int64_t>::max();
    max_interval_ = 0;
    num_misses_ = 0;
  }

  void tick() {
    const int64_t tick_ts = getTickTimestamp();
    if (last_tick_ts_ > 0) {
      const int64_t interval = tick_ts - last_tick_ts_;
      if (std::abs(interval - target_interval_) > tolerance_) {
        num_misses_++;
      }
      min_interval_ = std::min(min_interval_, interval);
      max_interval_ = std::max(max_interval_, interval);
    }
    last_tick_ts_ = tick_ts;
  }

  void stop() {
    const double miss_percentage =
        static_cast<double>(num_misses_) / static_cast<double>(getTickCount());
    EXPECT_LT(miss_percentage, get_tolerance_miss_percentage());
    LOG_INFO("min - expected - max: %lld - %lld - %lld", min_interval_, target_interval_,
             max_interval_);
    LOG_INFO("Miss percentage: %f %%", miss_percentage * 100.0);
  }

  ISAAC_PARAM(int, tolerance_dt, 200000);
  ISAAC_PARAM(double, tolerance_miss_percentage, 0.05);

 private:
  int64_t target_interval_;
  int64_t tolerance_;
  int64_t last_tick_ts_;
  int64_t min_interval_;
  int64_t max_interval_;
  int64_t num_misses_;
};

TEST(Alice, CodeletTickPeriodMonitor) {
  Application app;
  Node* node = app.createNode("test");
  CodeletTickPeriodMonitor* monitor = node->addComponent<CodeletTickPeriodMonitor>();
  monitor->async_set_tick_period("200Hz");
  app.startWaitStop(2.0);
  EXPECT_NEAR(monitor->getTickCount(), 400, 20);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::CodeletTickPeriodMonitor);
