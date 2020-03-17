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

TEST(Alice, Modules) {
  Application app(serialization::LoadJsonFromText(R"???(
{
  "name": "modules_test",
  "modules": [
    "message_generators",
    "flatsim"
  ],
  "graph": {
    "nodes": [
      {
        "name": "test",
        "components": [
          {
            "name": "0",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "1",
            "type": "isaac::message_generators::CameraGenerator"
          },
          {
            "name": "2",
            "type": "isaac::flatsim::DifferentialBasePhysics"
          }
        ]
      }
    ]
  }
})???"));
  EXPECT_NE(app.findComponentByName("test/0"), nullptr);
  EXPECT_NE(app.findComponentByName("test/1"), nullptr);
  EXPECT_NE(app.findComponentByName("test/2"), nullptr);
}

}  // namespace alice
}  // namespace isaac
