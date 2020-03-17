/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Behavior.hpp"

#include <string>
#include <unordered_set>

#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/alice/behaviors/Behavior.hpp"
#include "engine/alice/behaviors/NodeGroup.hpp"

namespace isaac {
namespace alice {
namespace behaviors {

namespace {

// Channel name used for status updates
const char kStatusEventChannel[] = "__status";

}  // namespace

void Behavior::stop() {
  for (size_t i = 0; i < getNumChildren(); i++) {
    Node& node = getChildByIndex(i);
    if (getChildStatus(node) == Status::RUNNING) {
      stopChild(node);
    }
  }
}

void Behavior::tickOnChildStatus() {
  std::unordered_set<std::string> triggers;
  for (size_t i = 0; i < getNumChildren(); i++) {
    getChildByIndex(i).iterateComponents([&](Component* component) {
      const std::string trigger = component->full_name() + "/" + kStatusEventChannel;
      triggers.insert(trigger);
    });
  }
  tickOnEvents(triggers);
}

size_t Behavior::getNumChildren() const {
  return children().getNumNodes();
}

Node& Behavior::getChildByIndex(size_t index) const {
  return children().getNodeByIndex(index);
}

Node* Behavior::findChildByName(const std::string& name) const {
  return children().findNodeByName(name);
}

Node& Behavior::getChildByName(const std::string& name) const {
  return children().getNodeByName(name);
}

Status Behavior::getChildStatus(Node& node) const {
  Status combined = Status::SUCCESS;
  node.iterateComponents([&](Component* component) {
    combined = Combine(combined, component->getStatus());
  });
  return combined;
}

void Behavior::startChild(Node& other) const {
  node()->app()->backend()->node_backend()->startNode(&other);
}

void Behavior::stopChild(Node& other) const {
  node()->app()->backend()->node_backend()->stopNode(&other);
}

NodeGroup& Behavior::children() const {
  if (children_ == nullptr) {
    children_ = node()->getComponent<NodeGroup>();
  }
  return *children_;
}

}  // namespace behaviors
}  // namespace alice
}  // namespace isaac
