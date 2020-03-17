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

enum class TestStatus { None=0, Constr, Start, Tick, Stop, Destr };

TestStatus status = TestStatus::None;
int tick_count = 0;

class MyCodelet : public Codelet {
 public:
  MyCodelet() {
    EXPECT_EQ(status, TestStatus::None);
    status = TestStatus::Constr;
  }
  void start() override {
    EXPECT_EQ(status, TestStatus::Constr);
    status = TestStatus::Start;
    tickPeriodically();
  }
  void tick() override {
    EXPECT_EQ(status, tick_count == 0 ? TestStatus::Start : TestStatus::Tick);
    status = TestStatus::Tick;
    tick_count++;
  }
  void stop() override {
    EXPECT_EQ(status, TestStatus::Tick);
    status = TestStatus::Stop;
  }
  ~MyCodelet() {
    EXPECT_EQ(status, TestStatus::Stop);
    status = TestStatus::Destr;
  }
};

TEST(Alice, SimpleCodelet) {
  EXPECT_EQ(status, TestStatus::None);
  {
    Application app;
    Node* node = app.createNode("test");
    auto* codelet = node->addComponent<MyCodelet>("simple");
    codelet->async_set_tick_period("50ms");
    app.startWaitStop(0.10);
  }
  EXPECT_EQ(status, TestStatus::Destr);
  EXPECT_GT(tick_count, 1);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet);
