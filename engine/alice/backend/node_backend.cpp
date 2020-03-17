/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "node_backend.hpp"

#include <algorithm>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/application_json_loader.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/codelet_backend.hpp"
#include "engine/alice/backend/component_registry.hpp"
#include "engine/alice/backend/config_backend.hpp"
#include "engine/alice/backend/modules.hpp"
#include "engine/alice/backend/names.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/components/Config.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/components/Pose.hpp"
#include "engine/core/logger.hpp"
#include "engine/gems/serialization/json.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

namespace {

constexpr std::chrono::seconds kMaxEmptyQueueWaitTime = std::chrono::seconds(1);

}  // namespace

NodeBackend::NodeBackend(Application* app, CodeletBackend* codelet_backend)
    : app_(app), codelet_backend_(codelet_backend) {}

Node* NodeBackend::createNode(const std::string& name) {
  // make sure that the name does not contains forbidden characters
  AssertValidName(name);
  // make sure the node does not exist yet
  {
    std::lock_guard<std::mutex> lock1(nodes_mutex_);
    const auto result = node_names_.insert(name);
    ASSERT(result.second, "A node with name '%s' already exists.", name.c_str());
  }
  LOG_DEBUG("Creating node '%s'", name.c_str());
  // Create a new node
  Node* node = Node::BackendAccess::ConstructNode(this, name);
  Node::BackendAccess::SetStage(node, Node::Stage::kConstructed);
  // The creation order matters
  node->addComponent<Config>();
  node->addComponent<Pose>();
  node->addComponent<Sight>();
  // Return owning pointer
  {
    std::lock_guard<std::mutex> lock1(nodes_mutex_);
    nodes_.emplace_back(node);
  }
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[node];
  }
  return node;
}

Node* NodeBackend::createMessageNode(const std::string& name) {
  Node* node = createNode(name);
  node->addComponent<MessageLedger>();
  return node;
}

void NodeBackend::destroyNode(const std::string& name) {
  Node* node;
  {
    // Check that the node exists (at the moment)
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    node = findNodeByNameImpl(name);
    ASSERT(node != nullptr, "Node with name '%s' does not exist");
    // Remove the node from the nodes list. This means that the node is not discoverable anymore.
    // The lambda we create below will take care of the remaining steps to destroy the node.
    auto it = std::find_if(nodes_.begin(), nodes_.end(),
        [node](auto& other) { return other.get() == node; });
    // We need to do a bit of an unusual operation here. We want to remove the unique pointer
    // holding the node from the node list. Ideally we would like to give this pointer to our lambda
    // below. However C++ standard does not allow to move a non-copyable lambda into an
    // std::function. Thus we release the unique pointer here and manually delete the node in the
    // lambda below.
    it->release();
    nodes_.erase(it);
  }
  // Add a job to destroy the job.
  std::lock_guard<std::mutex> lock(queue_mutex_);
  start_stop_queue_.emplace_back([this, node] {
    // Stop, destroy, and deconstruct the node
    stopImmediate(node);
    destroyImmediate(node);
    delete node;
  });
}

std::unique_ptr<Component> NodeBackend::createComponent(Node* node, const std::string& type_name,
                                                        std::string name) {
  // make sure that the name does not contains forbidden characters
  AssertValidName(name);
  // create the component
  std::unique_ptr<Component> uptr(app_->backend()->module_manager()->createComponent(type_name));
  uptr->name_ = std::move(name);
  uptr->uuid_ = Uuid::Generate();
  createComponentImpl(node, uptr.get());
  return uptr;
}

void NodeBackend::createNodeFromJson(const nlohmann::json& node_json, const Prefix& prefix) {
  static const std::vector<std::string> kDefaultComponents{
      "isaac::alice::Config", "isaac::alice::Pose", "isaac::alice::Sight"};
  static const std::string kMessageLeger = "isaac::alice::MessageLedger";
  static const std::string kNodeGroup = "isaac::alice::behaviors::NodeGroup";
  // get node name
  auto maybe_name = serialization::TryGetFromMap<std::string>(node_json, "name");
  ASSERT(maybe_name, "Missing mandatory field 'name' (type string) for node");
  const std::string node_name = prefix.apply(*maybe_name);
  // create a new node
  Node* node = createNode(node_name);
  // set disable_automatic_start if present in JSON
  const auto disable_automatic_start = app()->backend()->config_backend()->
      getAllForComponent(node_name, "disable_automatic_start");
  if (!disable_automatic_start.empty()) {
    // if set in config use this
    node->disable_automatic_start = disable_automatic_start;
  } else {
    // otherwise use check if set in graph
    auto disable_automatic_start_it = node_json.find("disable_automatic_start");
    if (disable_automatic_start_it != node_json.end()) {
      node->disable_automatic_start = *disable_automatic_start_it;
    }
  }

  // set start_order if present in JSON
  auto start_order_it = node_json.find("start_order");
  if (start_order_it != node_json.end()) {
    node->start_order = *start_order_it;
  }
  // find components to add
  auto node_components_json = node_json["components"];
  ASSERT(!node_components_json.is_null(), "Node must have entry 'components'");
  std::map<std::string, std::string> components_to_add;
  std::optional<std::string> message_ledger;
  std::optional<std::string> node_group;
  for (const auto& cjson : node_components_json) {
    // get the name
    auto name_json = cjson["name"];
    ASSERT(name_json.is_string(), "Component entry must contain 'name' of type string: %s",
           cjson.dump(2).c_str());
    const std::string name = name_json;
    // get the type
    auto type_json = cjson["type"];
    ASSERT(type_json.is_string(), "Component entry must contain 'type' of type string: %s",
           cjson.dump(2).c_str());
    const std::string type = type_json;
    // Don't add mandatory components.
    if (std::find(kDefaultComponents.begin(), kDefaultComponents.end(), type) !=
        kDefaultComponents.end()) {
      LOG_WARNING("Default components should not be explicitely specified: node = %s, "
                  "component = %s", node_name.c_str(), name.c_str());
      continue;
    }
    // Is it a message ledger?
    if (type == kMessageLeger) {
      ASSERT(!message_ledger, "Can only add one message ledger per node");
      message_ledger = name;
      continue;
    }
    // Is it a node group?
    if (type == kNodeGroup) {
      ASSERT(!node_group, "Can only add one node group per node");
      node_group = name;
      continue;
    }
    // Store for later
    auto it = components_to_add.find(name);
    ASSERT(it == components_to_add.end(), "Component name must be unique '%s'", name.c_str());
    components_to_add[name] = type;
  }
  // create components. make sure to create message ledger first
  if (message_ledger) {
    node->addComponent(kMessageLeger, *message_ledger);
  }
  if (node_group) {
    node->addComponent(kNodeGroup, *node_group);
  }
  for (const auto& kvp : components_to_add) {
    node->addComponent(kvp.second, kvp.first);
  }
}

Node* NodeBackend::findNodeByName(const std::string& name) const {
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  return findNodeByNameImpl(name);
}

void NodeBackend::start() {
  stop_requested_ = false;
  queue_running_ = true;

  // ToDo: If there is not a default blocker group this will fail.
  scheduler::JobDescriptor job_descriptor;
  job_descriptor.priority = 0;
  job_descriptor.execution_mode = scheduler::ExecutionMode::kBlockingOneShot;
  job_descriptor.name = "NodeBackend start/stop queue";
  job_descriptor.action = [this] { queueMain(); };
  job_handle_ = app()->backend()->scheduler()->createJobAndStart(job_descriptor);
  ASSERT(job_handle_, "Unable to create Node Queue Processing Job");

  // Sort nodes by start order
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  std::sort(nodes_.begin(), nodes_.end(),
            [](const auto& lhs, const auto& rhs) { return lhs->start_order < rhs->start_order; });
  // Start all nodes for which automatic start is not disabled
  std::lock_guard<std::mutex> lock2(queue_mutex_);
  for (auto& node_uptr : nodes_) {
    Node* node = node_uptr.get();
    if (node->disable_automatic_start) {
      continue;
    }
    addToStartStopQueue([this, node] { startImmediate(node); });
  }
}

void NodeBackend::stop() {
  stop_requested_ = true;
  {
    // TODO Can not use stopNodes here because we have unique_ptr in nodes_
    std::lock_guard<std::mutex> lock(nodes_mutex_);
    std::lock_guard<std::mutex> lock2(queue_mutex_);
    // Stop nodes in reverse order as a best effort
    for (auto it = nodes_.rbegin(); it != nodes_.rend(); ++it) {
      Node* node = it->get();
      if (node->disable_automatic_start) {
        continue;
      }
      addToStartStopQueue([this, node] { stopImmediate(node); });
    }
  }
  queue_running_ = false;
  // Extra notify for the case where no nodes are being destroyed at shutdown.
  queue_cv_.notify_all();
  app()->backend()->scheduler()->destroyJobAndWait(*job_handle_);
}

void NodeBackend::destroy() {
  std::lock_guard<std::mutex> lock(nodes_mutex_);
  for (auto& node : nodes_) {
    destroyImmediate(node.get());
  }
  nodes_.clear();
}

void NodeBackend::startNode(Node* node) {
  if (stop_requested_) {
    LOG_WARNING("Node '%s' not started because NodeBackend is shutting down", node->name().c_str());
    return;
  }
  std::lock_guard<std::mutex> lock(queue_mutex_);
  // TODO Check that the node is actually part of this application
  addToStartStopQueue([this, node] { startImmediate(node); });
}

void NodeBackend::stopNode(Node* node) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  // TODO Check that the node is actually part of this application
  addToStartStopQueue([this, node] { stopImmediate(node); });
}

void NodeBackend::startNodes(std::vector<Node*> nodes) {
  std::sort(nodes.begin(), nodes.end(),
            [](const auto& lhs, const auto& rhs) { return lhs->start_order < rhs->start_order; });
  for (Node* node : nodes) {
    startNode(node);
  }
}

void NodeBackend::stopNodes(const std::vector<Node*>& nodes) {
  for (Node* node : nodes) {
    stopNode(node);
  }
}

void NodeBackend::startComponent(Component* component) {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  // TODO Check that the component is actually part of this application
  addToStartStopQueue([this, component] { startImmediate(component); });
}

Node* NodeBackend::findNodeByNameImpl(const std::string& name) const {
  Node* node = nullptr;
  for (const auto& ptr : nodes_) {
    if (ptr->name() == name) {
      ASSERT(node == nullptr, "Found more than one node with name '%s'", name.c_str());
      node = ptr.get();
    }
  }
  return node;
}

void NodeBackend::startImmediate(Node* node) {
  const std::string& name = node->name();
  ASSERT(node, "argument null");
  if (node->getStage() != Node::Stage::kConstructed && node->getStage() != Node::Stage::kStopped) {
    LOG_WARNING("Can not start node '%s' because it is not in the stage 'constructed' or 'stopped'",
                name.c_str());
    return;
  }
  LOG_DEBUG("Starting node '%s'", name.c_str());
  Node::BackendAccess::SetStage(node, Node::Stage::kPreStart);
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[node].lifecycle = Lifecycle::kBeforeStart;
    statistics_[node].num_started++;
  }
  startComponents(node);
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[node].lifecycle = Lifecycle::kAfterStart;
  }
  Node::BackendAccess::SetStage(node, Node::Stage::kStarted);
}

void NodeBackend::stopImmediate(Node* node) {
  const std::string& name = node->name();
  ASSERT(node, "argument null");
  if (node->getStage() == Node::Stage::kStopped) {
    return;
  }
  if (node->getStage() != Node::Stage::kStarted) {
    LOG_WARNING("Can not stop node '%s' because it is not in the stage 'started'", name.c_str());
    return;
  }
  LOG_DEBUG("Stopping node '%s'", name.c_str());
  Node::BackendAccess::SetStage(node, Node::Stage::kPreStopped);
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[node].lifecycle = Lifecycle::kBeforeStop;
  }
  stopComponents(node);
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_[node].lifecycle = Lifecycle::kAfterStop;
  }
  Node::BackendAccess::SetStage(node, Node::Stage::kStopped);
}

void NodeBackend::destroyImmediate(Node* node) {
  ASSERT(node != nullptr, "Node must not be null");
  if (node->getStage() != Node::Stage::kStopped && node->getStage() != Node::Stage::kConstructed) {
    LOG_WARNING(
        "Can not destruct node '%s' because it is not in the stage 'stopped' or 'constructed'",
        node->name().c_str());
    return;
  }
  destroyComponents(node);
  {
    std::lock_guard<std::mutex> lock(statistics_mutex_);
    statistics_.erase(node);
  }
  Node::BackendAccess::SetStage(node, Node::Stage::kDestructed);
}

void NodeBackend::createComponentImpl(Node* node, Component* component) {
  ASSERT(node != nullptr, "Node must not be null");
  ASSERT(component != nullptr, "Component must not be null");
  ASSERT(node->findComponentByName(component->name()) == nullptr,
         "Component with name '%s' already exists", component->name().c_str());
  component->node_ = node;
  component->connectHooks();
  // We need special treatment for the config component.
  Config* config = dynamic_cast<Config*>(component);
  const bool is_config_component = config != nullptr;
  if (!is_config_component) {
    config = &node->config();
  }
  // In case this is the config component we need to initialize it first otherwise the backend
  // link will not be setup. In all other cases we need to intialize the component after we
  // have updated the hooks.
  if (is_config_component) {
    component->initialize();
  }
  // Set the type name of the component in the config.
  config->async_set(component, "__type_name", component->type_name());
  // Update configuration from root JSON
  config->updateHooks(component);
  // Make sure the config is aware of default parameters.
  config->updateCentralFromHooks(component);
  // Initialize the component either directly or initialize codelets via the backend.
  if (!component->is<Codelet>()) {
    if (!is_config_component) {
      component->initialize();
    }
  } else {
    codelet_backend_->initialize(static_cast<Codelet*>(component));
  }
}

void NodeBackend::addToStartStopQueue(std::function<void()> callback) {
  start_stop_queue_.push_back(callback);
  queue_cv_.notify_all();
}

void NodeBackend::startComponents(Node* node) {
  // Start components until no more component are added
  size_t next_unstarted_component_index = 0;
  while (true) {
    const std::vector<Component*> components = node->getComponents();
    // Reset the status to clear information from previous runs
    for (auto* component : components) {
      component->status_ = Status::RUNNING;
    }
    // First everything except codelets
    for (size_t i = next_unstarted_component_index; i < components.size(); i++) {
      auto* component = components[i];
      if (!component->is<Codelet>()) {
        component->start();
      }
    }
    // Second all codelets
    for (size_t i = next_unstarted_component_index; i < components.size(); i++) {
      auto* component = components[i];
      if (component->is<Codelet>()) {
        codelet_backend_->start(static_cast<Codelet*>(component));
      }
    }
    // Check if new components showed up while we were starting
    next_unstarted_component_index = components.size();
    Node::BackendAccess::SynchronizeComponentList(node);
    if (next_unstarted_component_index == node->getComponentCount()) {
      break;
    }
  }
}

void NodeBackend::stopComponents(Node* node) {
  std::vector<Component*> components = node->getComponents();
  // First all codelets (reverse order)
  for (auto it = components.rbegin(); it != components.rend(); ++it) {
    Component* component = *it;
    if (component->is<Codelet>()) {
      codelet_backend_->stop(static_cast<Codelet*>(component));
    }
  }
  // Second everything except codelets (reverse order)
  for (auto it = components.rbegin(); it != components.rend(); ++it) {
    Component* component = *it;
    if (!component->is<Codelet>()) {
      component->stop();
    }
  }
}

void NodeBackend::destroyComponents(Node* node) {
  const std::vector<Component*> components = node->getComponents();
  // First all codelets
  for (auto* component : components) {
    if (component->is<Codelet>()) {
      codelet_backend_->deinitialize(static_cast<Codelet*>(component));
    }
  }
  // Second everything except codelets
  for (auto* component : components) {
    if (!component->is<Codelet>()) {
      component->deinitialize();
    }
  }
}

void NodeBackend::startImmediate(Component* component) {
  if (!component->is<Codelet>()) {
    component->start();
  } else {
    codelet_backend_->start(static_cast<Codelet*>(component));
  }
}

void NodeBackend::queueMain() {
  while (queue_running_) {
    processQueue();
    std::unique_lock<std::mutex> lock(queue_mutex_);
    while (queue_running_ && start_stop_queue_.empty()) {
      // limit the amount of time we wait, so we can recheck the queue running flag
      queue_cv_.wait_for(lock, kMaxEmptyQueueWaitTime);
    }
  }
  // At this point all globally governed nodes are in the stop queue. However other nodes might
  // only be stopped during the stop of another node. So we iterate until the start/stop queue
  // is really empty.
  while (true) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (start_stop_queue_.empty()) {
        break;
      }
    }
    processQueue();
  }
}

void NodeBackend::processQueue() {
  std::vector<std::function<void()>> copy;
  {
    std::unique_lock<std::mutex> lock(queue_mutex_);
    copy = std::move(start_stop_queue_);
    start_stop_queue_ = {};
  }
  for (auto& task : copy) {
    task();
  }
}

nlohmann::json NodeBackend::getStatistics() const {
  std::lock_guard<std::mutex> lock2(nodes_mutex_);
  std::lock_guard<std::mutex> lock1(statistics_mutex_);
  nlohmann::json json;
  for (const auto& kvp : statistics_) {
    json[kvp.first->name()]["lifecycle"] = kvp.second.lifecycle;
    json[kvp.first->name()]["num_started"] = kvp.second.num_started;
  }
  return json;
}

}  // namespace alice
}  // namespace isaac
