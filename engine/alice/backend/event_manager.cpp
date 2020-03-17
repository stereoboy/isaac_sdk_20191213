/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "event_manager.hpp"

#include <string>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/core/logger.hpp"
#include "engine/gems/scheduler/scheduler.hpp"

namespace isaac {
namespace alice {

EventManager::EventManager(Application* app)
: app_(app) {}

void EventManager::onStatusUpdate(Component* component) {
  const std::string event = component->full_name() + "/__status";
  const int64_t target_time = app_->backend()->clock()->timestamp();
  app_->backend()->scheduler()->notify(event, target_time);

  // stop the node if one failed or all succeeded
  Node* node = component->node();
  const auto node_status = node->getStatus();
  if (node_status != Status::RUNNING) {
    if (node_status == Status::SUCCESS) {
      LOG_DEBUG("Stopping node '%s' because it reached status '%s'",
                node->name().c_str(), ToString(node_status));
    } else {
      LOG_ERROR("Stopping node '%s' because it reached status '%s'",
                node->name().c_str(), ToString(node_status));
    }
    app_->backend()->node_backend()->stopNode(node);
  }
}

}  // namespace alice
}  // namespace isaac
