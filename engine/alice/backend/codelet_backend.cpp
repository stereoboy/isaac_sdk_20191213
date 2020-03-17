/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "codelet_backend.hpp"

#include <cmath>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/components/Pose.hpp"
#include "engine/alice/components/Scheduling.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/sight/sight.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

namespace {

// Rounds a number to at most two digits after the point
double Round2(double x) {
  return std::round(x * 100.0) / 100.0;
}

}  // namespace

class CodeletBackend::CodeletSchedule {
 public:
  CodeletSchedule(Codelet* codelet)
      : codelet_(codelet), is_started_(false), id_(codelet_->full_name()) {
    ASSERT(codelet_, "Codelet must not be null");
  }

  ~CodeletSchedule() {
    ASSERT(!is_started_, "Codelet '%s' was not stopped after it was started", id().c_str());
  }

  std::optional<scheduler::JobHandle> job_handle;

  bool isStarted() const { return is_started_; }
  Codelet* codelet() const { return codelet_; }
  // An identifier string containing node and component name
  const std::string& id() const { return id_; }

  void start() {
    ASSERT(!is_started_, "Codelet '%s' already started", id().c_str());
    LOG_DEBUG("Starting codelet '%s' ...", id().c_str());

    ledger_ = codelet_->node()->getComponentOrNull<MessageLedger>();

    // Prepare for start
    codelet_->node()->config().updateHooks(codelet_);
    codelet_->onBeforeStart();

    codelet_->start();

    is_started_ = true;
    LOG_DEBUG("Starting codelet '%s' DONE", id().c_str());
  }

  void tick() {
    ASSERT(is_started_, "Codelet '%s' was not started", id().c_str());

    // Do not tick codelets which are not running anymore.
    if (codelet_->getStatus() != Status::RUNNING) return;

    // Update messages for receiving hooks and only tick if there is something to tick off.
    if (!updateRx()) {
      return;
    }

    // Prepare for tick
    codelet_->node()->config().updateDirtyHooks(codelet_);
    codelet_->onBeforeTick();

    codelet_->tick();
  }

  void stop() {
    ASSERT(is_started_, "Codelet '%s' was not started", id().c_str());
    LOG_DEBUG("Stopping codelet '%s' ...", id().c_str());

    // Prepare for stop
    codelet_->node()->config().updateDirtyHooks(codelet_);
    codelet_->onBeforeStop();

    codelet_->stop();

    is_started_ = false;
    LOG_DEBUG("Stopping codelet '%s' DONE", id().c_str());
  }

  Codelet* codelet_;
  bool is_started_;
  MessageLedger* ledger_;

 private:
  bool updateRx() {
    // A codelet is considered to be ready for execution either when it ticks periodically and
    // does not have any triggers, or if the trigger check was disabled explicitely.
    bool is_triggered = codelet_->triggers_.empty() || codelet_->non_rx_triggered_;
    if (ledger_ == nullptr) {
      return is_triggered;
    }
    for (auto& kvp : codelet_->rx_hook_trackers_) {
      auto& tracker = kvp.second;
      ASSERT(tracker.rx, "Encountered null pointer");
      ASSERT(tracker.rx->component() == codelet_, "RX component is not identical to this codelet");
      // find all new messages and pass them to all synchronizers
      // if there is no synchronizer, then just put the message directly into the hook
      ledger_->checkChannelMessages(
          {tracker.rx->component(), tracker.rx->tag()},  // endpoint
          [&](const std::vector<ConstMessageBasePtr>& messages) {
            for (const auto& message : messages) {
              ASSERT(message, "Encountered null pointer");
              if (!tracker.last_timestamp || message->pubtime > *tracker.last_timestamp) {
                tracker.last_timestamp = message->pubtime;
                // give message to synchronizer
                bool found_sync = false;
                for (auto& sync : codelet_->synchronizers_) {
                  if (sync->contains(tracker.rx->tag())) {
                    sync->push(tracker.rx->tag(), message);
                    found_sync = true;
                  }
                }
                if (!found_sync) {
                  // set message directly
                  tracker.rx->setMessage(message);
                  is_triggered = true;
                }
              }
            }
          });
    }
    // check synchronizers for new message
    std::map<std::string, alice::ConstMessageBasePtr> messages;
    for (const auto& sync : codelet_->synchronizers_) {
      while (sync->sync_pop(messages)) {
        for (const auto& kvp : messages) {
          codelet_->rx_hook_trackers_[kvp.first].rx->setMessage(kvp.second);
          is_triggered = true;
        }
      }
    }
    return is_triggered;
  }

  const std::string id_;
};

CodeletBackend::CodeletBackend() {
  // for unique_ptr destructor
}

CodeletBackend::~CodeletBackend() {
  // for unique_ptr destructor
}

void CodeletBackend::start() {}

void CodeletBackend::stop() {}

void CodeletBackend::onChangeTicking(Codelet* codelet) {
  ASSERT(codelet != nullptr, "Codelet must not be null");
  std::shared_lock<std::shared_timed_mutex> lock(codelets_mutex_);
  auto it = codelets_.find(codelet->uuid());
  ASSERT(it != codelets_.end(), "Codelet was not added");
  if (it->second->job_handle) {
    app()->backend()->scheduler()->destroyJobAndWait(*(it->second->job_handle));
    it->second->job_handle = std::nullopt;
  }
  if (it->second->isStarted()) {
    addToScheduler(it->second.get());
  }
}

void CodeletBackend::initialize(Codelet* codelet) {
  ASSERT(codelet != nullptr, "Codelet must not be null");
  codelet->backend_ = this;
  codelet->initialize();
  CodeletSchedule* program = new CodeletSchedule(codelet);
  {
    std::unique_lock<std::shared_timed_mutex> lock(codelets_mutex_);
    codelets_[codelet->uuid()] = std::unique_ptr<CodeletSchedule>(program);
  }
}

void CodeletBackend::start(Codelet* codelet) {
  ASSERT(codelet != nullptr, "Codelet must not be null");
  CodeletSchedule* codelet_schedule;
  {
    std::shared_lock<std::shared_timed_mutex> lock(codelets_mutex_);
    auto it = codelets_.find(codelet->uuid());
    ASSERT(it != codelets_.end(), "Codelet was not added");
    codelet_schedule = it->second.get();
  }
  codelet_schedule->start();
  addToScheduler(codelet_schedule);
}

void CodeletBackend::stop(Codelet* codelet) {
  ASSERT(codelet != nullptr, "Codelet must not be null");
  std::shared_lock<std::shared_timed_mutex> lock(codelets_mutex_);
  auto it = codelets_.find(codelet->uuid());
  ASSERT(it != codelets_.end(), "Codelet was not added");
  CodeletSchedule* codelet_schedule = it->second.get();
  if (codelet_schedule->job_handle) {
    app()->backend()->scheduler()->destroyJobAndWait(*(it->second->job_handle));
    codelet_schedule->job_handle = std::nullopt;
  }
  codelet_schedule->stop();
}

void CodeletBackend::deinitialize(Codelet* codelet) {
  ASSERT(codelet != nullptr, "Codelet must not be null");
  std::shared_lock<std::shared_timed_mutex> lock(codelets_mutex_);
  auto it = codelets_.find(codelet->uuid());
  ASSERT(it != codelets_.end(), "Codelet was not added");
  codelets_.erase(it);
  codelet->deinitialize();
}

void CodeletBackend::addToScheduler(CodeletSchedule* program) {
  Codelet* codelet = program->codelet();
  scheduler::JobDescriptor job_descriptor;
  job_descriptor.name = program->id();

  // Set the scheduling parameters
  Scheduling* scheduling = codelet->node()->getComponentOrNull<Scheduling>();
  if (scheduling) {
    job_descriptor.slack = SecondsToNano(scheduling->get_slack());
    if (auto deadline = scheduling->try_get_deadline()) {
      job_descriptor.deadline = SecondsToNano(*deadline);
    }
    job_descriptor.priority = scheduling->get_priority();
    job_descriptor.execution_group = scheduling->get_execution_group();
  } else {
    job_descriptor.slack = 0;
    job_descriptor.priority = 0;
    job_descriptor.execution_group = "";
  }

  // Execute the tick of the codelet as an action
  job_descriptor.action = [program] { program->tick(); };

  if (!codelet->triggers_.empty()) {
    job_descriptor.execution_mode = scheduler::ExecutionMode::kEventTask;
    // Always allow one extra event to trigger so we don't miss work.
    job_descriptor.event_trigger_limit = codelet->get_execution_queue_limit();
  } else if (codelet->tick_period_ == 0) {
    job_descriptor.execution_mode = scheduler::ExecutionMode::kBlocking;
  } else if (codelet->tick_period_ > 0) {
    job_descriptor.execution_mode = scheduler::ExecutionMode::kPeriodicTask;
    job_descriptor.period = codelet->tick_period_;
    if (!job_descriptor.deadline) {
      job_descriptor.deadline = codelet->tick_period_;
    } else if (job_descriptor.deadline && job_descriptor.deadline > codelet->tick_period_) {
      LOG_WARNING("Deadline parameter set greater than tick period. Tick period used instead.");
      job_descriptor.deadline = codelet->tick_period_;
    }
  } else {
    LOG_WARNING("Codelet '%s' was not added to scheduler because no tick method is specified.",
                program->id().c_str());
    return;
  }
  program->job_handle = app()->backend()->scheduler()->createJob(job_descriptor);
  if (program->job_handle) {
    if (!codelet->triggers_.empty()) {
      app()->backend()->scheduler()->registerEvents(*(program->job_handle), codelet->triggers_);
    }
    app()->backend()->scheduler()->startJob(*(program->job_handle));
  }
  LOG_DEBUG("Starting job for codelet '%s'", program->id().c_str());
}

nlohmann::json CodeletBackend::getStatistics() const {
  std::shared_lock<std::shared_timed_mutex> lock(codelets_mutex_);
  const double time = ToSeconds(app()->backend()->clock()->timestamp());
  nlohmann::json json;
  for (const auto& kvp : codelets_) {
    if (!kvp.second->job_handle) {
      // Ignore un-scheduled jobs
      continue;
    }
    auto stats = app()->backend()->scheduler()->getJobStatistics(*(kvp.second->job_handle));
    stats.current_load.updateTime(time);
    stats.current_rate.updateTime(time);
    auto* codelet = kvp.second->codelet();
    nlohmann::json codelet_json;
    codelet_json["average_exec_dt"] = Round2(1000.0 * stats.exec_dt.value());
    codelet_json["average_late_dt"] = Round2(1000.0 * stats.getAverageOverrunTime());
    codelet_json["late_p"] = Round2(100.0 * stats.getOverrunPercentage());
    codelet_json["load"] = Round2(100.0 * stats.current_load.rate());
    codelet_json["frequency"] = Round2(stats.current_rate.rate());
    codelet_json["dt"] = Round2(1000.0 * codelet->getTickDt());
    codelet_json["num_ticks"] = codelet->getTickCount();
    codelet_json["median"] = Round2(1000.0 * stats.execution_time_median.median());
    codelet_json["p90"] = Round2(1000.0 * stats.execution_time_median.percentile(0.9));
    codelet_json["max"] = Round2(1000.0 * stats.execution_time_median.max());
    json[codelet->node()->name()][codelet->name()] = std::move(codelet_json);
  }
  return json;
}

}  // namespace alice
}  // namespace isaac
