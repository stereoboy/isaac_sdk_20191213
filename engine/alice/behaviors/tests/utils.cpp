/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "utils.hpp"

#include <string>

#include "engine/alice/alice.hpp"
#include "engine/alice/behaviors/ConstantBehavior.hpp"
#include "engine/alice/behaviors/TimerBehavior.hpp"

namespace isaac {
namespace alice {

behaviors::ConstantBehavior* CreateConstantBehaviorNode(Application& app, const std::string& name,
                                                        Status status) {
  auto* behavior = CreateSubBehaviorNode<behaviors::ConstantBehavior>(app, name);
  behavior->async_set_status(status);
  return behavior;
}

behaviors::TimerBehavior* CreateTimerBehaviorNode(Application& app, const std::string& name,
                                                  double delay, Status status) {
  auto* behavior = CreateSubBehaviorNode<behaviors::TimerBehavior>(app, name);
  behavior->async_set_delay(delay);
  behavior->async_set_status(status);
  return behavior;
}

}  // namespace alice
}  // namespace isaac

