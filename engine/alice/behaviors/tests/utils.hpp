/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>
#include <vector>

#include "engine/alice/alice.hpp"
#include "engine/alice/behaviors/ConstantBehavior.hpp"
#include "engine/alice/behaviors/NodeGroup.hpp"
#include "engine/alice/behaviors/TimerBehavior.hpp"

namespace isaac {
namespace alice {

// Creates a group node with a behavior of given type
template <typename Behavior>
Behavior* CreateSubBehaviorNode(Application& app, const std::string& name) {
  Node* node = app.createNode(name);
  node->disable_automatic_start = true;
  return node->addComponent<Behavior>();
}

// Creates a group node with a behavior of given type
template <typename Behavior>
Behavior* CreateCompositeBehaviorNode(Application& app, const std::string& name,
                                      const std::vector<std::string>& children) {
  Node* node = app.createNode(name);
  auto* node_group = node->addComponent<behaviors::NodeGroup>();
  node_group->async_set_node_names(children);
  return node->addComponent<Behavior>();
}

// Creates a node with a behavior returning the given status
behaviors::ConstantBehavior* CreateConstantBehaviorNode(Application& app, const std::string& name,
                                                        Status status);

// Creates a node with a timer behavior
behaviors::TimerBehavior* CreateTimerBehaviorNode(Application& app, const std::string& name,
                                                  double delay, Status status);

}  // namespace alice
}  // namespace isaac

