/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>

#include <vector>

#include "engine/alice/backend/component_registry.hpp"
#include "engine/alice/backend/lifecycle.hpp"
#include "engine/alice/node.hpp"
#include "engine/gems/scheduler/job_descriptor.hpp"
#include "engine/gems/serialization/json.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

class Application;
class Component;
class CodeletBackend;
class Node;
class Prefix;

// Handles node and component lifetime
class NodeBackend {
 public:
  // Statistics about a node
  struct NodeStatistics {
    // The current lifecycle of the node
    Lifecycle lifecycle = Lifecycle::kNeverStarted;
    // The number of times the node was started
    int num_started = 0;
  };

  NodeBackend(Application* app, CodeletBackend* codelet_backend);

  // The app to which this node backend belongs
  Application* app() const { return app_; }

  // Creates a new node. `name` is user-defined unique name for the node. This function will assert
  // if the name is not unique.
  Node* createNode(const std::string& name);
  // Like `createNode` and also adds message passing capabilities
  Node* createMessageNode(const std::string& name);

  // Destroys a node and all its components.
  void destroyNode(const std::string& name);

  // Finds a node by name. Will return nullptr if no node with this name exists.
  Node* findNodeByName(const std::string& name) const;

  // Returns the number of nodes
  size_t numNodes() const { return nodes_.size(); }

  // Returns a list of all nodes
  std::vector<Node*> nodes() const {
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::vector<Node*> result(nodes_.size());
    std::transform(nodes_.begin(), nodes_.end(), result.begin(),
                   [](const auto& uptr) { return uptr.get(); });
    return result;
  }

  // Creates a new component of given type and name in a node.
  std::unique_ptr<Component> createComponent(Node* node, const std::string& type_name,
                                             std::string name);

  // Loads nodes from JSON
  // The top-level structure is an array of nodes where each node is a dictionary. Each node has
  // certain fields and among them is one called 'components'. This is an array itself where
  // each component again is a dictionary with certain fields.
  // An example:
  //   {
  //     "name": "myname",
  //     "components": [
  //       {"name": "comp1", "type": "type1"},
  //       {"name": "comp2", "type": "type2"},
  //     ]
  //   },
  void createNodeFromJson(const nlohmann::json& node_json, const Prefix& prefix);

  // Starts the backend and all nodes which where added so far
  void start();
  // Stops the backend and all nodes
  void stop();
  // Destroys the backend and all nodes
  void destroy();

  // Starts a node and all its components
  void startNode(Node* node);
  // Stops a node and all its components
  void stopNode(Node* node);
  // Starts a list of nodes. Starting nodes multiple times is a warning.
  void startNodes(std::vector<Node*> nodes);
  void startNodes(std::initializer_list<Node*> nodes) {
    // TODO implement this more elegantly without a memory allocation
    startNodes(std::vector<Node*>{nodes});
  }
  // Stops a list of nodes nodes. Starting nodes multiple times is a warning.
  void stopNodes(const std::vector<Node*>& nodes);
  void stopNodes(std::initializer_list<Node*> nodes) {
    // TODO implement this more elegantly without a memory allocation
    stopNodes(std::vector<Node*>{nodes});
  }

  // Starts a component in case it was added after the node was already started. The component
  // will not be started in this thread.
  void startComponent(Component* component);

  Node* findNodeByNameImpl(const std::string& name) const;

  // Gets statistics about nodes
  nlohmann::json getStatistics() const;

 private:
  void createComponentImpl(Node* node, Component* component);
  void addToStartStopQueue(std::function<void()> callback);

  void startImmediate(Node* node);
  void stopImmediate(Node* node);
  void destroyImmediate(Node* node);

  void startComponents(Node* node);
  void stopComponents(Node* node);
  void destroyComponents(Node* node);

  // Starts a component immeditaly in this thread.
  void startImmediate(Component* component);

  void queueMain();
  void processQueue();

  Application* app_ = nullptr;
  CodeletBackend* codelet_backend_;

  mutable std::mutex nodes_mutex_;
  mutable std::condition_variable queue_cv_;
  std::set<std::string> node_names_;
  std::vector<std::unique_ptr<Node>> nodes_;

  bool stop_requested_;

  // Use scheduler
  std::mutex queue_mutex_;
  std::optional<scheduler::JobHandle> job_handle_;
  std::vector<std::function<void()>> start_stop_queue_;
  std::atomic<bool> queue_running_;

  mutable std::mutex statistics_mutex_;
  std::map<Node*, NodeStatistics> statistics_;
};

}  // namespace alice
}  // namespace isaac
