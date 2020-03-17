/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Behavior.hpp"

#include <string>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/behavior_backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

void Behavior::initialize() {
  backend_ = node()->app()->backend()->getBackend<BehaviorBackend>();
  // Load the children from config
  if (!parseLinksFromConfig()) {
    LOG_ERROR("Could not parse behavior configuration");
    return;
  }
  backend_->registerComponent(this);
}

void Behavior::deinitialize() {
  backend_->unregisterComponent(this);
  backend_ = nullptr;
}

Node* Behavior::get(const std::string& alias) {
  const auto it = child_nodes_.find(alias);
  if (it == child_nodes_.end()) {
    LOG_ERROR("Could not find child node with alias '%s'", alias.c_str());
    return nullptr;
  }
  return it->second;
}

bool Behavior::stop(const std::string& alias) {
  Node* child = get(alias);
  if (!child) {
    LOG_ERROR("Could not stop behavior with alias '%s'", alias.c_str());
    return false;
  }
  node()->app()->backend()->node_backend()->stopNode(child);
  return true;
}

bool Behavior::start(const std::string& alias) {
  Node* child = get(alias);
  if (!child) {
    LOG_ERROR("Could not start behavior with alias '%s'", alias.c_str());
    return false;
  }
  node()->app()->backend()->node_backend()->startNode(child);
  return true;
}

bool Behavior::parseLinksFromConfig() {
  const auto link_aliases = get_aliases();
  const auto link_names = get_nodes();
  const size_t n = link_names.size();
  if (link_aliases.size() != n) {
    LOG_ERROR("alias (%zu) and links (%zu) must have the same size", link_aliases.size(), n);
    return false;
  }
  for (size_t i = 0; i < n; i++) {
    const auto& alias = link_aliases[i];
    const auto& name = link_names[i];
    Node* child = node()->app()->findNodeByName(name);
    if (child == nullptr) {
      LOG_ERROR("Could not find node '%s'", name.c_str());
      return false;
    }
    const auto result = child_nodes_.insert({alias, child});
    if (!result.second) {
      LOG_ERROR("Could not insert into map");
      return false;
    }
  }
  return true;
}

}  // namespace alice
}  // namespace isaac
