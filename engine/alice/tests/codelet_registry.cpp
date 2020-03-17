/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(CodeletRegistry, Builtin) {
  auto names = Singleton<ComponentRegistry>::Get().getNames();
  EXPECT_EQ(names.count("isaac::alice::Config"), 1);
}

class MyCodelet1 : public Codelet { };
class MyCodelet2 : public Codelet { };

TEST(CodeletRegistry, Codelets) {
  auto names = Singleton<ComponentRegistry>::Get().getNames();
  EXPECT_EQ(names.count("isaac::alice::MyCodelet1"), 1);
  EXPECT_EQ(names.count("isaac::alice::MyCodelet2"), 1);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet1);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet2);
