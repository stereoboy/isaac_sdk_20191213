/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>
#include <memory>
#include <shared_mutex>  // NOLINT
#include <string>

#include "engine/alice/backend/component_backend.hpp"
#include "engine/alice/components/Codelet.hpp"
#include "engine/gems/scheduler/scheduler.hpp"
#include "engine/gems/serialization/json.hpp"

namespace isaac {
namespace alice {

// Manages codelet execution
class CodeletBackend : public ComponentBackend<Codelet> {
 public:
  CodeletBackend();
  ~CodeletBackend();

  void start() override;
  void stop() override;

  // Notifies the backend that the tick schedule of a codelet changed
  void onChangeTicking(Codelet* codelet);

  // Gets statistics about codelets
  nlohmann::json getStatistics() const;

 protected:
  // Calls codelet initialize and does additional setup
  void initialize(Codelet* codelet);
  // Does some bookkeeping to start the codelet and calls codelet start
  void start(Codelet* codelet);
  // Does some bookkeeping to stop the codelet and calls codelet stop
  void stop(Codelet* codelet);
  // Calls codelet deinitialize and does some additional cleanup
  void deinitialize(Codelet* codelet);

 private:
  friend class Backend;
  friend class NodeBackend;

  class CodeletSchedule;

  void addToScheduler(CodeletSchedule* codelet);

  mutable std::shared_timed_mutex codelets_mutex_;
  std::map<Uuid, std::unique_ptr<CodeletSchedule>> codelets_;
};

}  // namespace alice
}  // namespace isaac
