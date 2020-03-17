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

class MyVariableWriter : public Codelet {
 public:
  void start() override {
    tickPeriodically();
    setVariable("hallo", getTickTime(), 0.1);
  }
  void tick() override {
    setVariable("hallo", getTickTime(), 0.2);
  }
};

class MyVariableReader : public Codelet {
 public:
  void start() override {
    tickPeriodically();
  }
  void tick() override {
    auto maybe = getVariable("writer/isaac.alice.MyVariableWriter/hallo", getTickTime());
    ASSERT_NE(maybe, std::nullopt);
    EXPECT_EQ(*maybe, 0.2);
  }
};

TEST(Alice, Config) {
  Application app;

  Node* writer_node = app.createNode("writer");
  auto* writer = writer_node->addComponent<MyVariableWriter>();
  writer->async_set_tick_period("10ms");

  Node* reader_node = app.createNode("reader");
  auto* reader = reader_node->addComponent<MyVariableReader>();
  reader->async_set_tick_period("10ms");

  app.startWaitStop(0.30);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyVariableWriter);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyVariableReader);
