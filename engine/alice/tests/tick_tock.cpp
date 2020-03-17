/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class TickTock : public Codelet {
 public:
  void start() override {
    tickPeriodically();
    count = 0;
  }
  void tick() override {
    if (count % 2 == 0) {
      LOG_INFO("tick");
    } else {
      LOG_INFO("tock");
    }
    count ++;
  }
  void stop() override {
    LOG_INFO("Ticked %d times", count);
  }
  int count;
};

TEST(TickTock, Test) {
  Application app;
  Node* publisher = app.createNode("ticktock");
  TickTock* tick_tock = publisher->addComponent<TickTock>();
  tick_tock->async_set_tick_period("100ms");
  app.startWaitStop(0.55);
  EXPECT_NEAR(tick_tock->count, 6, 1);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::TickTock);
