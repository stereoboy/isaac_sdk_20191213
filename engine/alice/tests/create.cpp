/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <set>

#include "engine/alice/alice.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyCodelet : public Codelet { };

TEST(Alice, CreateApp) {
  Application app(nlohmann::json{{"name", "alice_tests_create_app"}});
  EXPECT_EQ(app.name(), "alice_tests_create_app");
  app.startWaitStop(0.01);
}

TEST(Alice, CreateAppUuid) {
  constexpr int kNumIterations = 20;
  std::set<Uuid> uuids;
  std::set<std::string> names;
  for (int i=0; i<kNumIterations; i++) {
    Application app;
    names.insert(app.name());
    uuids.insert(app.uuid());
    app.startWaitStop(0.01);
  }
  EXPECT_EQ(names.size(), kNumIterations);
  EXPECT_EQ(uuids.size(), kNumIterations);
}

TEST(Alice, CreateNode) {
  Application app;
  Node* node = app.createNode("spacestation");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->name(), "spacestation");
  Node* node2 = app.createNode("mars");
  ASSERT_NE(node2, nullptr);
  EXPECT_EQ(node2->name(), "mars");
  app.startWaitStop(0.01);
}

TEST(Alice, CreateDuplicateNode) {
  Application app;
  Node* node = app.createNode("spacestation");
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->name(), "spacestation");
  EXPECT_DEATH(app.createNode("spacestation"), ".");
  app.startWaitStop(0.01);
}

TEST(Alice, CreateNodeWithComponent) {
  Application app;
  Node* node = app.createNode("test");
  auto* codelet = node->addComponent<MyCodelet>();
  ASSERT_NE(codelet, nullptr);
  app.startWaitStop(0.01);
}

TEST(Alice, CreateNodeWithComponentForbiddenName) {
  Application app;
  EXPECT_DEATH(app.createNode("oh/no"), ".?may not contain.?");
  Node* node = app.createNode("test");
  EXPECT_DEATH(node->addComponent<MyCodelet>("rea/ly"), ".?may not contain.?");
  EXPECT_EQ(node->addComponent<MyCodelet>("test")->name(), "test");
}

TEST(Alice, ComponentNames) {
  Application app;
  Node* node = app.createNode("test");
  auto* codelet = node->addComponent<MyCodelet>();
  ASSERT_NE(codelet, nullptr);
  EXPECT_EQ(node->findComponentByName("isaac.alice.MyCodelet"), codelet);
  EXPECT_EQ(node->findComponentByName("second"), nullptr);
  EXPECT_DEATH(codelet = node->addComponent<MyCodelet>(), ".?exists.?");
  auto* codelet2 = node->addComponent<MyCodelet>("second");
  ASSERT_NE(codelet2, nullptr);
  EXPECT_EQ(codelet2->name(), "second");
  EXPECT_EQ(node->findComponentByName("second"), codelet2);
  EXPECT_EQ(node->findComponentByName("first"), nullptr);
  app.startWaitStop(0.01);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet);
