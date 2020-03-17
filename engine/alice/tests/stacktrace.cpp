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

class MyCrash : public Codelet {
 public:
  void start() {
    PANIC("help");
  }
};

TEST(Alice, Stacktrace) {
  Application app;
  app.createNode("error")->addComponent<MyCrash>("error");
  EXPECT_DEATH(app.startWaitStop(0.01), ".*terminated unexpectedly.*#01.*Minidump.*");
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCrash);
