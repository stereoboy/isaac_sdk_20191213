/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>
#include <vector>

#include "engine/alice/status.hpp"
#include "engine/core/logger.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

class Hook;
class Node;
class NodeBackend;
class Pose;

// Components are the essential basic building blocks of every node
class Component {
 public:
  virtual ~Component() {}

  // The node which this component is part of
  Node* node() const { return node_; }

  // Called to initialize a component.
  // This function should be called after the component has been constructed in the sense of c++,
  // but before 'start' is called or any any work is done by the component. It is recommended
  // that this function not do a significant amount of work.
  virtual void initialize() {}
  // Called to start the component
  // This function is where the bulk of the work for putting the component in a running state
  // should be done. Called before any work is done.
  virtual void start() {}
  // Called to stop the component
  // This function is the natural counter point to 'start' in terms of work that should be done.
  // Any tear down work that is necessary when the component is  being stopped should be done here.
  virtual void stop() {}
  // Called to deinitialize the component
  // The natural counter point to initialize this function should be called before c++ destruction
  // to undo anything done by initialize if necessary.
  virtual void deinitialize() {}

  // The name of the base type of this component, e.g. Codelet
  const std::string& base_type_name() const { return base_name_; }
  // The name of the type of this component, e.g. Config
  const std::string& type_name() const { return type_name_; }
  // The name of the component
  const std::string& name() const { return name_; }
  // This value can be used to uniquly identify each component across all applications
  const Uuid& uuid() const { return uuid_; }
  // The full name of the component including the node
  std::string full_name() const;

  // Returns true if this component is of the given type.
  template <typename T>
  bool is() const {
    return dynamic_cast<const T*>(this) != nullptr;
  }
  // Returns this component as its actual type if it is of the given type, or null otherwise.
  template <typename T>
  T* as() const {
    return dynamic_cast<const T*>(this);
  }

  // Adds a hook to the component
  void addHook(Hook* hook) { hooks_.push_back(hook); }
  // Internal function called after the component is added to the node
  void connectHooks();

  // Returns a list with all hooks associated with this component
  const std::vector<Hook*>& hooks() const { return hooks_; }

  // Marks this component as successfull
  void reportSuccess() { reportSuccess(""); }
  void reportSuccess(const char* message, ...) {
    va_list args;
    va_start(args, message);
    updateStatusImpl(Status::SUCCESS, message, args);
    va_end(args);
  }
  // Marks this component as failed
  void reportFailure() { reportFailure(""); }
  void reportFailure(const char* message, ...) {
    va_list args;
    va_start(args, message);
    updateStatusImpl(Status::FAILURE, message, args);
    va_end(args);
  }
  // Updates the status of the component
  void updateStatus(Status new_status) { updateStatus(new_status, ""); }
  void updateStatus(Status new_status, const char* message, ...);
  // Gets the current status of the comopnent
  Status getStatus() const { return status_; }
  // Gets the message associated with the last status update
  std::string getStatusMessage() const { return status_message_; }

 private:
  friend class NodeBackend;
  friend class ComponentRegistry;

  // Implementation of updateStatus
  void updateStatusImpl(Status new_status, const char* message, va_list args);

  std::string base_name_;
  std::string type_name_;
  std::string name_;
  Uuid uuid_;
  Node* node_ = nullptr;

  // To connect hooks
  std::vector<Hook*> hooks_;

  // Run status
  Status status_ = Status::RUNNING;
  std::string status_message_;
};

}  // namespace alice
}  // namespace isaac
