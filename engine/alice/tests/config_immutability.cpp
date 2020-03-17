/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

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

// A helper class for tests. It uses a configuration parameter many times during tick.
class ConfigUser : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }

  void tick() override {
    const int first = get_value();
    for (int i = 0; i < 500; i++) {
      Sleep(1'000);  // TODO Need a time-safe sleep
      ASSERT_EQ(get_value(), first);
    }
    const int next = 1234;
    set_value(next);
    for (int i = 0; i < 500; i++) {
      Sleep(1'000);  // TODO Need a time-safe sleep
      ASSERT_EQ(get_value(), next);
    }
  }

  void stop() override {
    EXPECT_GT(getTickCount(), 0);
  }

  ISAAC_PARAM(int, value, 0);
};

// A helper class for tests. It changes a configuration parameter of another codelet many times
// during tick.
class ConfigModifier : public Codelet {
 public:
  void start() override {
     user_ = node()->app()->findComponentByName<ConfigUser>(get_link());
    tickPeriodically();
  }

  void tick() override {
    for (int i = 0; i < 1000; i++) {
      user_->async_set_value(i);
      Sleep(1'000);  // TODO Need a time-safe sleep
    }
  }

  ISAAC_PARAM(std::string, link);

 private:
  ConfigUser* user_;
};

TEST(Config, ThreadSafety) {
  // Test that a codelet using a configuration parameter during its tick is guaranteed to not
  // observe changes to that configuration parameter which are issued from outside of the codelet.
  Application app;
  for (int i=0; i<10; i++) {
    const std::string user_name = "user_" + std::to_string(i);
    ConfigUser* user = app.createNode(user_name)->addComponent<ConfigUser>("user");
    user->async_set_tick_period("100 Hz");
    ConfigModifier* mod =
        app.createNode("modifer_" + std::to_string(i))->addComponent<ConfigModifier>();
    mod->async_set_tick_period("100 Hz");
    mod->async_set_link(user_name + "/user");
  }
  app.startWaitStop(2.00);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ConfigUser);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ConfigModifier);
