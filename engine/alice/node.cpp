/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "node.hpp"

#include <regex>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/core/assert.hpp"

namespace isaac {
namespace alice {

Application* Node::app() {
  ASSERT(backend_->app(), "argument null");
  return backend_->app();
}

Clock* Node::clock() {
  ASSERT(backend_->app()->backend()->clock(), "argument null");
  return backend_->app()->backend()->clock();
}

size_t Node::getComponentCount() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  return components_.size();
}

std::vector<Component*> Node::getComponents() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  std::vector<Component*> components(components_.size());
  for (size_t i = 0; i < components.size(); i++) {
    components[i] = components_[i].get();
  }
  return components;
}

void Node::iterateComponents(std::function<void(Component*)> callback) {
  std::unique_lock<std::mutex> lock(component_mutex_);
  for (const auto& node : components_) {
    callback(node.get());
  }
}

Component* Node::findComponentByName(const std::string& name) const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  for (const auto& uptr : components_) {
    if (uptr->name() == name) {
      return uptr.get();
    }
  }
  return nullptr;
}

bool Node::hasComponentByName(const std::string& name) const {
  return findComponentByName(name) != nullptr;
}

Component* Node::addComponent(const std::string& type_name, std::string name) {
  auto uptr = backend_->createComponent(this, type_name, std::move(name));
  std::unique_lock<std::mutex> lock(component_mutex_);
  Component* ptr = uptr.get();
  if (stage_ == Stage::kConstructed) {
    // We can add the component directly to the component list as the start process has not yet
    // been started.
    components_.emplace_back(std::move(uptr));
  } else if (stage_ == Stage::kPreStart) {
    // Add the component not directly to the component list, but instead store them for later.
    components_added_while_starting_.emplace_back(std::move(uptr));
  } else if (stage_ == Stage::kStarted) {
    components_.emplace_back(std::move(uptr));
    // When a component is added to a running node we need to start the component immediately.
    // Otherwise we delay it until the node is started which will then also start all components.
    backend_->startComponent(ptr);
  } else {
    PANIC("It is not allowed to add a component (typename: '%s', name: '%s') to a node (name '%s' "
          ") in its current state ('%d'). ", type_name.c_str(), ptr->name().c_str(),
          this->name().c_str(), static_cast<int>(stage_));
  }
  return ptr;
}

std::string Node::ComputeNameFromType(const std::string& type) {
  // Replace "::"" from fully qualified C++ typenames with '.' to create a readable name
  return std::regex_replace(type, std::regex("::"), ".");
}

void Node::synchronizeComponentList() {
  std::unique_lock<std::mutex> lock(component_mutex_);
  components_.insert(components_.end(),
                     make_move_iterator(components_added_while_starting_.begin()),
                     make_move_iterator(components_added_while_starting_.end()));
  components_added_while_starting_.clear();
}

}  // namespace alice
}  // namespace isaac
