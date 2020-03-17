/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "scheduler.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/core/assert.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/scheduler/clock.hpp"
#include "engine/gems/scheduler/job.hpp"
#include "engine/gems/scheduler/time_machine.hpp"

namespace isaac {
namespace scheduler {

constexpr char Scheduler::kDefaultBlockerGroup[];
constexpr char Scheduler::kDefaultWorkerGroup[];

namespace {
// Default scheduler group names. To be used when
// a execution group is not provided by the user.
constexpr char kDefaultWorkerGroup[] = "__WorkerGroup__";
constexpr char kDefaultBlockerGroup[] = "__BlockerGroup__";
}  // namespace

Scheduler::Scheduler(const std::vector<ExecutionGroupDescriptor>& execution_groups, Clock* clock)
    : clock_(clock), is_running_(false), handle_counter_(0) {
  time_machine_ = std::make_unique<TimeMachine>(clock);
  for (auto& group : execution_groups) {
    execution_groups_.emplace(std::piecewise_construct, std::forward_as_tuple(group.name),
                              std::forward_as_tuple(group, clock_, time_machine_.get(),
                              &handle_counter_));
  }
  createDefaultExecutionGroups();
  logExecutionGroups();
  int total_workers = 0;
  for (auto& group : execution_groups_) {
    if (group.second.description.has_workers) {
      total_workers = group.second.description.cores.size();
      time_machine_->registerExecutionGroup(&(group.second));
    }
  }
  time_machine_->setTotalWorkerCount(total_workers);
}

Scheduler::~Scheduler() {
  stop();
}

std::optional<JobHandle> Scheduler::createJob(const JobDescriptor& descriptor) {
  std::string group_name = descriptor.execution_group;
  if (group_name.empty()) {
    switch (descriptor.execution_mode) {
      case ExecutionMode::kBlocking:
      case ExecutionMode::kBlockingOneShot:
        group_name = kDefaultBlockerGroup;
        break;
      case ExecutionMode::kEventTask:
      case ExecutionMode::kPeriodicTask:
      case ExecutionMode::kOneShotTask:
        group_name = kDefaultWorkerGroup;
        break;
      default:
        LOG_ERROR("Invalid execution mode. Unable to create a job");
        return std::nullopt;
    }
  }
  auto* group = findExecutionGroup(group_name);
  if (!group) {
    LOG_ERROR("Unable to create job for execution group %s. No such execution group.",
              group_name.c_str());
    return std::nullopt;
  }
  JobHandle handle = group->createJob(descriptor);
  {
    std::lock_guard<std::mutex> lock(job_map_mutex_);
    job_to_execution_group_[handle] = group_name;
  }
  return handle;
}

void Scheduler::destroyJob(const JobHandle& handle) {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }
  group->destroyJob(handle);
}

void Scheduler::startJob(const JobHandle& handle) const {
  auto group_name = findExecutionGroupName(handle);
  ASSERT(group_name, "Unable to start job: Handle %ull", handle);
  auto* group = findExecutionGroup(*group_name);
  ASSERT(group, "Unable to start job: Handle %ull", handle);
  group->startJob(handle);
}

void Scheduler::stopJob(const JobHandle& handle) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }
  group->stopJob(handle);
}

void Scheduler::waitForJobDestruction(const JobHandle& handle) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }
  group->waitForJobDestruction(handle);
}

void Scheduler::waitForJobStop(const JobHandle& handle) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }
  group->waitForJobStop(handle);
}

std::optional<JobHandle> Scheduler::createJobAndStart(const JobDescriptor& descriptor) {
  std::optional<JobHandle> handle = createJob(descriptor);
  if (handle) {
    startJob(*handle);
  }
  return handle;
}

void Scheduler::stopJobAndWait(const JobHandle& handle) {
  stopJob(handle);
  waitForJobStop(handle);
}
void Scheduler::destroyJobAndWait(const JobHandle& handle) {
  destroyJob(handle);
  waitForJobDestruction(handle);
}

void Scheduler::registerEvents(const JobHandle& handle,
                               const std::unordered_set<std::string>& events) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }

  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }

  group->registerEvents(handle, events);
}

void Scheduler::unregisterEvents(const JobHandle& handle,
                                 const std::unordered_set<std::string>& events) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return;
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return;
  }
  group->unregisterEvents(handle, events);
}

void Scheduler::notify(const std::string& event, int64_t target_time) const {
  for (auto& group : execution_groups_) {
    group.second.notify(event, target_time);
  }
}

void Scheduler::start() {
  is_running_ = true;

  // setup and start all execution groups
  for (auto& group : execution_groups_) {
    group.second.start();
  }
}

void Scheduler::stop() {
  if (!is_running_.exchange(false)) {
    return;
  }
  for (auto& group : execution_groups_) {
    group.second.stop();
  }
  std::lock_guard<std::mutex> lock(job_map_mutex_);
  job_to_execution_group_.clear();
}

void Scheduler::startWaitStop(double duration) {
  start();
  Sleep(SecondsToNano(duration));
  stop();
}

JobStatistics Scheduler::getJobStatistics(const JobHandle& handle) const {
  auto group_name = findExecutionGroupName(handle);
  if (!group_name) {
    return JobStatistics{};
  }
  auto* group = findExecutionGroup(*group_name);
  if (!group) {
    return JobStatistics{};
  }
  return group->getJobStatistics(handle);
}

std::vector<JobStatistics> Scheduler::getJobStatistics() const {
  std::lock_guard<std::mutex> lock(job_map_mutex_);
  std::vector<JobStatistics> all_stats;
  for (const auto& kvp : execution_groups_) {
    const auto stats = kvp.second.getJobStatistics();
    all_stats.insert(all_stats.end(), stats.begin(), stats.end());
  }
  return all_stats;
}

double Scheduler::getExecutionDelay() const {
  double max_delay = -1;
  for (auto& group : execution_groups_) {
    double group_delay = group.second.getExecutionDelay();
    if (max_delay < group_delay) {
      max_delay = group_delay;
    }
  }
  return max_delay;
}

void Scheduler::enableTimeMachine() {
  time_machine_->start();
}
void Scheduler::disableTimeMachine() {
  time_machine_->stop();
}

std::optional<std::string> Scheduler::findExecutionGroupName(const JobHandle& handle) const {
  std::optional<std::string> group_name;
  {
    std::lock_guard<std::mutex> lock(job_map_mutex_);
    auto it = job_to_execution_group_.find(handle);
    if (it != job_to_execution_group_.end()) {
      group_name = it->second;
    }
  }
  return group_name;
}

ExecutionGroup* Scheduler::findExecutionGroup(const std::string& group_name) const {
  auto it = execution_groups_.find(group_name);
  if (it != execution_groups_.end()) {
    return &(it->second);
  }
  return nullptr;
}

void Scheduler::logExecutionGroups() const {
  LOG_INFO("Scheduler execution groups are:");
  for (const auto& group : execution_groups_) {
    const auto& description = group.second.description;
    std::string cores_str;
    for (size_t i = 0; i < description.cores.size(); i++) {
      cores_str += std::to_string(description.cores[i])
          + (i + 1 < description.cores.size() ? ", " : "");
    }
    LOG_INFO("%s: Cores = [%s], Workers = %s", description.name.c_str(), cores_str.c_str(),
             description.has_workers ? "Yes" : "No");
  }
}

void Scheduler::createDefaultExecutionGroups() {
  // Check if default groups are required.
  bool need_default_worker = true;
  bool need_default_blocker = true;
  std::string worker_name(kDefaultWorkerGroup);
  std::string blocker_name(kDefaultBlockerGroup);

  for (auto& group : execution_groups_) {
    if (worker_name.compare(group.second.description.name) == 0) {
      need_default_worker = false;
    }
    if (blocker_name.compare(group.second.description.name) == 0) {
      need_default_blocker = false;
    }
  }

  if (need_default_blocker || need_default_worker) {
    std::unordered_set<unsigned int> used_cores;
    // Count all currently used cores
    for (auto& group : execution_groups_) {
      for (auto core : group.second.description.cores) {
        used_cores.insert(core);
      }
    }
    unsigned int num_cores = std::thread::hardware_concurrency();
    if (execution_groups_.empty()) {
      ASSERT(num_cores > 0, "Unable to determine a default scheduler configuration.");
    }
    unsigned int free_core_count =
        num_cores > used_cores.size() ? num_cores - used_cores.size() : 0;

    LOG_WARNING(
        "No default execution groups specified. Attempting to create scheduler configuration for "
        "%d remaining cores. This may be non optimal for the system and application.",
        free_core_count);

    ExecutionGroupDescriptor default_worker_group;
    default_worker_group.name = kDefaultWorkerGroup;
    default_worker_group.has_workers = true;

    ExecutionGroupDescriptor default_blocker_group;
    default_blocker_group.name = kDefaultBlockerGroup;
    default_blocker_group.has_workers = false;

    if (free_core_count == 0) {
      // There are jobs in the sdk which must have a place to execute for the system
      // to operate. If a user reserves all cores override their decisions slightly
      LOG_WARNING("All cores specified by config, all default work will be pinned to core %d.",
                  num_cores - 1);
      default_worker_group.cores.push_back(num_cores - 1);
      default_blocker_group.cores.push_back(num_cores - 1);
    } else if (free_core_count == 1) {
      // Find the free core and assign both workers and blockers to it.
      for (unsigned int i = 0; i < num_cores; ++i) {
        auto it = used_cores.find(i);
        if (it == used_cores.end()) {
          default_worker_group.cores.push_back(i);
          default_blocker_group.cores.push_back(i);
        }
      }
    } else {
      unsigned int num_workers = static_cast<unsigned int>(.75 * free_core_count);
      unsigned int num_blockers = free_core_count - num_workers;

      // Give 3/4 of the remaining cores to workers and 1/4 to blockers
      auto group = execution_groups_.find(kDefaultWorkerGroup);
      unsigned int index = 0, worker_core_count = 0, blocker_core_count = 0;
      for (; index < num_cores && worker_core_count < num_workers; ++index) {
        auto it = used_cores.find(index);
        if (it == used_cores.end()) {
          default_worker_group.cores.push_back(index);
          ++worker_core_count;
        }
      }
      group = execution_groups_.find(kDefaultBlockerGroup);
      for (; index < num_cores && blocker_core_count < num_blockers; ++index) {
        auto it = used_cores.find(index);
        if (it == used_cores.end()) {
          default_blocker_group.cores.push_back(index);
          ++blocker_core_count;
        }
      }
    }
    // Create a worker group and a blocker group
    if (need_default_worker) {
      execution_groups_.emplace(std::piecewise_construct,
                                std::forward_as_tuple(kDefaultWorkerGroup),
                                std::forward_as_tuple(default_worker_group, clock_,
                                                      time_machine_.get(), &handle_counter_));
    }
    if (need_default_blocker) {
      execution_groups_.emplace(std::piecewise_construct,
                                std::forward_as_tuple(kDefaultBlockerGroup),
                                std::forward_as_tuple(default_blocker_group, clock_,
                                                      time_machine_.get(), &handle_counter_));
    }
  }
}

}  // namespace scheduler
}  // namespace isaac
