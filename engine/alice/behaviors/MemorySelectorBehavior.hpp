/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/behaviors/Behavior.hpp"

namespace isaac {
namespace alice {
namespace behaviors {

// @experimental
// A behavior which will execute on child after another until one succeeds. The behavior will
// succeed once one of its children succeeded. If all children fail or if there are no children the
// selector behavior will fail.
// This behavior has a memory and remembers which child it is currently executing. This is a
// modification of the default selector behavior.
class MemorySelectorBehavior : public Behavior {
 public:
  void start() override;
  void tick() override;

 private:
  // The index of the currently active child.
  size_t current_child_index_;
};

}  // namespace behaviors
}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::behaviors::MemorySelectorBehavior);
