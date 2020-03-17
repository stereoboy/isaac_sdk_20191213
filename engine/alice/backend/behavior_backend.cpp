/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "behavior_backend.hpp"

#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

void BehaviorBackend::registerComponent(Behavior* behavior) {
  ASSERT(behavior, "Must not be a nullptr");
  // Check that all children are still available
  for (const auto& kvp : behavior->child_nodes_) {
    Node* child = kvp.second;
    auto it = node_to_parent_map_.find(child);
    if (it != node_to_parent_map_.end()) {
      LOG_ERROR("Could not add node '%s' as child to behavior '%s' as it is already the child of"
                " another behavior in node '%s'", child->name().c_str(),
                behavior->full_name().c_str(), it->second->name().c_str());
      behavior->child_nodes_.clear();
      return;
    }
  }
  // Mark child nodes
  for (const auto& kvp : behavior->child_nodes_) {
    Node* child = kvp.second;
    node_to_parent_map_[child] = behavior->node();
    child->disable_automatic_start = true;
  }
}

void BehaviorBackend::unregisterComponent(Behavior* behavior) {
  for (const auto& kvp : behavior->child_nodes_) {
    Node* child = kvp.second;
    node_to_parent_map_.erase(child);
    child->disable_automatic_start = false;
  }
}

}  // namespace alice
}  // namespace isaac
