/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class ConfigParallel : public Codelet {
 public:
  void start() override {
    counter_ = 0;
    tickPeriodically();
  }

  void tick() override {
    for (int i=0; i<100; i++) {
      node()->config().async_set(this, std::to_string(counter_), counter_);
      counter_++;
    }
  }

  void stop() override {
    EXPECT_NEAR(getTickCount(), 100, 50);
  }

 private:
  int counter_;
};

TEST(Config, Parallel) {
  Application app;
  std::vector<ConfigParallel*> messies;
  for (int i=0; i<20; i++) {
    messies.push_back(app.createNode("messy_" + std::to_string(i))->addComponent<ConfigParallel>());
    //Hardcode test values to avoid config messiness
    messies.back()->async_set_tick_period("10ms");
  }
  app.startWaitStop(1.00);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ConfigParallel);
