/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/core/time.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyCodelet : public Codelet { };

TEST(Alice, DestroyNode) {
  Application app(nlohmann::json{{"name", "destroy_node"}});
  app.createNode("hello");
  app.start();
  Sleep(SecondsToNano(0.1));
  ASSERT_NE(app.findNodeByName("hello"), nullptr);
  app.destroyNode("hello");
  LOG_ERROR("AA 1");
  Sleep(SecondsToNano(0.1));
  LOG_ERROR("AA 2");
  ASSERT_EQ(app.findNodeByName("hello"), nullptr);
  LOG_ERROR("AA 3");
  app.stop();
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet);
