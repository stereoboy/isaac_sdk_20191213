/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/backend/clock.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/components/Config.hpp"
#include "engine/alice/components/Pose.hpp"
#include "engine/alice/components/Sight.hpp"
#include "engine/core/assert.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

class Application;
class Clock;
class NodeBackend;

// An application consists of many nodes which each fulfills a certain purpose. For example
// localizing the robot or planning a path could be implemented via a node. Nodes are created out
// of various components which define the node's functionality.
class Node {
 public:
  // The lifecycle stage of the node
  enum class Stage {
    kInitial,      // The C++ constructor of the node was called.
    kConstructed,  // The node was created.
    kPreStart,     // The node is starting up but not all its components are started yet.
    kStarted,      // The node and all its components are started.
    kPreStopped,   // the node is stopping but not all its components are stopped yet.
    kStopped,      // The node and all its components are stopped.
    kDestructed    // The node is destroyed and the C++ destructor will be called.
  };

  // A class which provides access to certain private functionality only for the NodeBackend.
  class BackendAccess {
   private:
    // Only the NodeBackend has access.
    friend class NodeBackend;

    // Creates a node with the given name
    static Node* ConstructNode(NodeBackend* backend, std::string name) {
      Node* node = new Node();
      node->backend_ = backend;
      node->name_ = std::move(name);
      return node;
    }

    // See `Node::synchronizeComponentList`
    static void SynchronizeComponentList(Node* node) {
      node->synchronizeComponentList();
    }

    // See `Node::setStage`
    static void SetStage(Node* node, Stage stage) {
      node->setStage(stage);
    }
  };

  // The name of this node is useful for human readability but must not be unique
  const std::string& name() const { return name_; }

  // The app for this node
  Application* app();

  // The clock backend for this node
  Clock* clock();

  // Returns the current lifecycle stage of the node
  Stage getStage() const { return stage_; }
  // Returns true if the node was started
  bool isStarted() const { return stage_ == Stage::kStarted; }

  // Creates a new component of given type in a node. The component can be created with the given
  // name or otherwise the typename is used as the basis of the name. Components must have unique
  // names within a node, otherwise these functions will assert.
  template <typename T>
  T* addComponent();
  template <typename T>
  T* addComponent(std::string name);
  Component* addComponent(const std::string& type_name, std::string name);

  // Checks if this node has a component of the given type
  template <typename T>
  bool hasComponent() const;

  // Returns the number of components in the node
  size_t getComponentCount() const;
  // Gets a list of components
  std::vector<Component*> getComponents() const;

  // Calls a callback for every component. Do not execute a lot of work in the callback as this
  // function blocks until finished.
  void iterateComponents(std::function<void(Component*)> callback);

  // Gets all components with the given type
  template <typename T>
  std::vector<T*> getComponents() const;
  // Gets component with the given type
  // Asserts if there are multiple or no components of that type.
  template <typename T>
  T* getComponent() const;
  // Gets component with the given type, or null if no such component or multiple components
  template <typename T>
  T* getComponentOrNull() const;
  // Gets component with the given name, or null if no such component exists
  Component* findComponentByName(const std::string& name) const;
  // Returns true if this node has a component with this name
  bool hasComponentByName(const std::string& name) const;
  // Gets the component for the given type and if none exists creates one first
  template <typename T>
  T* getOrAddComponent();

  // Gets the config component of this node (this component always exists)
  Config& config() const { return getComponentCached(config_component_); }
  // Gets the pose component (this component always exists)
  Pose& pose() const { return getComponentCached(pose_component_); }
  // Gets the sight visualization component (this component always exists)
  Sight& sight() const { return getComponentCached(sight_component_); }

  // Gets the status of the node based on the status of its components
  Status getStatus() const {
    Status combined = Status::SUCCESS;
    for (const auto& x : components_) {
      combined = Combine(combined, x->getStatus());
    }
    return combined;
  }

  // Set this to true if the node should not be started automatically when the applications starts.
  // Note that nodes created at runtime are never started automatically and always need to be
  // started manually with app->backend()->node_backed()->start(my_node);
  bool disable_automatic_start = false;

  // The order in which nodes will be started. The smaller the earliers. Note that this of course
  // only applies to nodes which are started at the same time. If a node is started dynamically this
  // does not have any effect.
  int start_order = 0;

 private:
  // Derives a name for a component from its type
  static std::string ComputeNameFromType(const std::string& type);

  // Only NodeBackend can create instances
  Node() {}

  // Gets a component via a cache. This will get the component only once and from there one use
  // the cache instead.
  template <typename Component>
  Component& getComponentCached(Component*& component) const {
    if (component == nullptr) {
      component = getComponent<Component>();
    }
    return *component;
  }

  // Adds components which where added while the node was starting to the general set of components
  void synchronizeComponentList();

  // Changes the stage in which the node is
  void setStage(Stage stage) { stage_ = stage; }

  NodeBackend* backend_;

  Stage stage_ = Stage::kInitial;

  std::string name_;

  mutable std::mutex component_mutex_;

  std::vector<std::unique_ptr<Component>> components_;
  std::vector<std::unique_ptr<Component>> components_added_while_starting_;

  mutable Config* config_component_ = nullptr;
  mutable Pose* pose_component_ = nullptr;
  mutable Sight* sight_component_ = nullptr;
};

// Implementation

template <typename T>
T* Node::addComponent() {
  return addComponent<T>(ComputeNameFromType(ComponentName<T>::TypeName()));
}

template <typename T>
T* Node::addComponent(std::string name) {
  auto ptr = addComponent(ComponentName<T>::TypeName(), name);
  T* ptr_t = dynamic_cast<T*>(ptr);
  ASSERT(ptr_t, "Failed to create component of correct type.");
  return ptr_t;
}

template <typename T>
bool Node::hasComponent() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  for (const auto& uptr : components_) {
    T* ptr = dynamic_cast<T*>(uptr.get());
    if (ptr != nullptr) {
      return true;
    }
  }
  return false;
}

template <typename T>
std::vector<T*> Node::getComponents() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  std::vector<T*> result;
  for (const auto& uptr : components_) {
    T* ptr = dynamic_cast<T*>(uptr.get());
    if (ptr != nullptr) {
      result.push_back(ptr);
    }
  }
  return result;
}

template <typename T>
T* Node::getComponent() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  T* result = nullptr;
  for (const auto& uptr : components_) {
    T* ptr = dynamic_cast<T*>(uptr.get());
    if (ptr != nullptr) {
      ASSERT(result == nullptr,
          "Found multiple components of type '%s' in node '%s'. "
          "Use `getComponents` to get multiple components.",
          ComponentName<T>::TypeName(), name().c_str());
      result = ptr;
    }
  }
  ASSERT(result != nullptr,
      "Could not find a component of type '%s' in node '%s'. "
      "Use `hasComponent` or `getComponents` to check for components.",
      ComponentName<T>::TypeName(), name().c_str());
  return result;
}

template <typename T>
T* Node::getComponentOrNull() const {
  std::unique_lock<std::mutex> lock(component_mutex_);
  T* result = nullptr;
  for (const auto& uptr : components_) {
    T* ptr = dynamic_cast<T*>(uptr.get());
    if (ptr != nullptr) {
      if (result != nullptr) {
        return nullptr;
      }
      result = ptr;
    }
  }
  return result;
}

template <typename T>
T* Node::getOrAddComponent() {
  if (hasComponent<T>()) {
    return getComponent<T>();
  } else {
    return addComponent<T>();
  }
}

}  // namespace alice
}  // namespace isaac
