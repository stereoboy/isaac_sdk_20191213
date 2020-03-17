/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "component.hpp"

#include <string>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/event_manager.hpp"
#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

std::string Component::full_name() const {
  return node()->name() + "/" + name();
}

void Component::connectHooks() {
  for (Hook* hook : hooks_) {
    hook->connect();
  }
}

void Component::updateStatus(Status new_status, const char* message, ...) {
  ASSERT(new_status == Status::SUCCESS || new_status == Status::FAILURE,
         "updateStatus can only be called with SUCCESS or FAILURE");
  va_list args;
  va_start(args, message);
  updateStatusImpl(new_status, message, args);
  va_end(args);
}

void Component::updateStatusImpl(Status new_status, const char* message, va_list args) {
  // Check that the status is only updated when the component is alive
  const auto stage = node()->getStage();
  ASSERT(stage == Node::Stage::kPreStart || stage == Node::Stage::kStarted
         || stage == Node::Stage::kPreStopped , "reportSuccess or reportFailure can only be "
         "called in start, stop, or tick. component: %s", full_name().c_str());
  status_ = new_status;

  // Create the message
  va_list args2;
  va_copy(args2, args);
  std::vector<char> buffer(1 + std::vsnprintf(NULL, 0, message, args2), '\0');
  va_end(args2);
  std::vsnprintf(buffer.data(), buffer.size(), message, args);
  status_message_ = std::string(buffer.data(), buffer.size());

  // Print error messages to the console. We might want to revisit this later.
  if (new_status == Status::FAILURE) {
    LOG_ERROR("Component '%s' of type '%s' reported FAILURE:\n\n"
              "    %s\n", full_name().c_str(), type_name().c_str(), status_message_.c_str());
  }

  node()->app()->backend()->event_manager()->onStatusUpdate(this);
}

}  // namespace alice
}  // namespace isaac
