/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/backend/stopwatch.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, ClockCreated) {
  Application app;
  Node* node = app.createNode("test");
  ASSERT_NE(node, nullptr);
  ASSERT_NE(node->clock(), nullptr);
}

class MyTimeTestCodelet : public Codelet {
 public:
  void start() override {
    ASSERT_NE(node()->clock(), nullptr);

    tickPeriodically();
    num_ticks_ = 0;
  }
  void tick() override {
    num_ticks_ ++;
    ASSERT_EQ(num_ticks_ == 1, isFirstTick());
    if (isFirstTick()) {
      stopwatch();
      return;
    }
    EXPECT_NEAR(getTickDt(), expected_tick_period_, 0.001);
    EXPECT_NEAR(stopwatch().read(), expected_tick_period_, 0.005);
    stopwatch().start();
  }
  void stop() override {
    EXPECT_EQ(num_ticks_, getTickCount());
  }

  void setPeriod(int dt_ms) {
    expected_tick_period_ = 0.001 * dt_ms;
    set_tick_period(std::to_string(dt_ms) + "ms");
  }

 private:
  int num_ticks_;
  double expected_tick_period_;
};

TEST(Alice, ClockTicking) {
  Application app;
  app.createNode("test")->addComponent<MyTimeTestCodelet>()->setPeriod(27);
  app.startWaitStop(0.50);
}

TEST(Alice, ClockTickingDual) {
  Application app;
  app.createNode("ben")->addComponent<MyTimeTestCodelet>()->setPeriod(20);
  app.createNode("bertha")->addComponent<MyTimeTestCodelet>()->setPeriod(36);
  app.startWaitStop(0.50);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyTimeTestCodelet);
