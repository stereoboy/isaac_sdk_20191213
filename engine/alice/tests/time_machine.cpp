/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/scheduler/scheduler.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

// A simple codelet which sleeps a bit
class Tick : public Codelet {
 public:
  void start() override { tickPeriodically(); }
  void tick() override { Sleep(work_duration); }
  void stop() override {}
  int64_t work_duration;
};

void RunTest(int duration_ms, int period_ms) {
  constexpr double kAppDuration = 1.0;
  Application app;
  Node* tick_node = app.createNode("tick");
  Tick* tick = tick_node->addComponent<Tick>();
  tick->work_duration = int64_t{1'000'000} * static_cast<int64_t>(duration_ms);
  tick->async_set_tick_period(std::to_string(period_ms) + "ms");
  app.backend()->scheduler()->enableTimeMachine();
  app.startWaitStop(kAppDuration);
  const unsigned max_throughput = static_cast<int>(kAppDuration / ToSeconds(tick->work_duration));
  EXPECT_GT(tick->getTickCount(), 0.4 * max_throughput);
  // Sometimes one extra run is squeezed in as it can start just before shutdown
  // and the system is not premptive.
  EXPECT_LT(tick->getTickCount(), 2 * max_throughput);
  LOG_ERROR("%d ms | %d ms | actual throughput %d | max throughput %d | efficiency %f", duration_ms,
            period_ms, tick->getTickCount(), max_throughput,
            static_cast<double>(tick->getTickCount()) / static_cast<double>(max_throughput));
}

TEST(TimeMachine, Test) {
  RunTest(5, 100);
  RunTest(10, 100);
  RunTest(25, 100);
  RunTest(100, 100);
  RunTest(5, 10);
  RunTest(10, 10);
  RunTest(25, 10);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Tick);
