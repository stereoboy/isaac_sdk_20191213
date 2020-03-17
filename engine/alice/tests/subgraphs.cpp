/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/gems/serialization/json.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, SubGraphsIndependent) {
  auto maybe_json =
      serialization::TryLoadJsonFromFile("engine/alice/tests/subgraphs_independent.app.json");
  ASSERT_TRUE(maybe_json);
  Application app(*maybe_json);
  app.startWaitStop(0.55);
}

TEST(Alice, SubGraphsInterconnected) {
  auto maybe_json =
      serialization::TryLoadJsonFromFile("engine/alice/tests/subgraphs_interconnected.app.json");
  ASSERT_TRUE(maybe_json);
  Application app(*maybe_json);
  app.startWaitStop(0.55);
}

TEST(Alice, SubGraphsAndRegularNodes) {
  auto maybe_json =
      serialization::TryLoadJsonFromFile("engine/alice/tests/subgraphs_and_regular_nodes.app.json");
  ASSERT_TRUE(maybe_json);
  Application app(*maybe_json);
  app.startWaitStop(0.55);
}

TEST(Alice, SubGraphsInterfaced) {
  auto maybe_json =
      serialization::TryLoadJsonFromFile("engine/alice/tests/subgraphs_with_interfaces.app.json");
  ASSERT_TRUE(maybe_json);
  Application app(*maybe_json);
  app.startWaitStop(0.55);
}

TEST(Alice, SubGraphsNested) {
  auto maybe_json =
      serialization::TryLoadJsonFromFile("engine/alice/tests/subgraphs_nested.app.json");
  ASSERT_TRUE(maybe_json);
  Application app(*maybe_json);
  app.startWaitStop(0.55);
}

}  // namespace alice
}  // namespace isaac
