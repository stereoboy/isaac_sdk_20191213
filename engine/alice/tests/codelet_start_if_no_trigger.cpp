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

enum class TestStatus { None=0, Constr, Start, Stop, Destr };

TestStatus status = TestStatus::None;

class MyCodelet : public Codelet {
 public:
  MyCodelet() {
    EXPECT_EQ(status, TestStatus::None);
    status = TestStatus::Constr;
  }
  void start() override {
    EXPECT_EQ(status, TestStatus::Constr);
    status = TestStatus::Start;
  }
  void tick() override {
    EXPECT_FALSE(true);  // should never go here
  }
  void stop() override {
    EXPECT_EQ(status, TestStatus::Start);
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
    node->addComponent<MyCodelet>();
    app.startWaitStop(0.05);
  }
  EXPECT_EQ(status, TestStatus::Destr);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet);
