/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "allocator_backend.hpp"

#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/core/allocator/allocators.hpp"
#include "engine/core/allocator/cached_allocator.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/scheduler/scheduler.hpp"

namespace isaac {
namespace alice {

namespace {

// Time period for which the allocator will collect allocation statistics.
constexpr double kStatisticsCollectionDuration = 10.0;

// Number of buckets to use for cached allocator
constexpr int kNumBuckets = 128;

}  // namespace

AllocatorBackend::AllocatorBackend(Application* app)
: app_(app) { }

void AllocatorBackend::start() {
  // Adds a job which will switch the global allocator to cached mode after collecting allocation
  // statistics for a certain time period.
  scheduler::JobDescriptor job_descriptor;
  job_descriptor.priority = 0;
  job_descriptor.execution_mode = scheduler::ExecutionMode::kOneShotTask;
  job_descriptor.target_start_time = SecondsToNano(kStatisticsCollectionDuration);
  job_descriptor.name = "AllocatorBackend";
  job_descriptor.action = [this] {
    auto* cpu_allocator = dynamic_cast<CachedAllocator*>(GetCpuAllocator());
    if (cpu_allocator == nullptr) {
      LOG_ERROR("Could not switch to cached allocation mode as the global CPU allocator is not of "
                "type CachedAllocator");
    } else {
      cpu_allocator->switchToCachedMode(kNumBuckets);
      LOG_INFO("Optimized memory CPU allocator.");
    }

    auto* cuda_allocator = dynamic_cast<CachedAllocator*>(GetCudaAllocator());
    if (cuda_allocator == nullptr) {
      LOG_ERROR("Could not switch to cached allocation mode as the global CUDA allocator is not of "
                "type CachedAllocator");
    } else {
      cuda_allocator->switchToCachedMode(kNumBuckets);
      LOG_INFO("Optimized memory CUDA allocator.");
    }
  };
  app_->backend()->scheduler()->createJobAndStart(job_descriptor);

  start_time_ = NowCount();
}

void AllocatorBackend::stop() {
  if (auto* allocator = dynamic_cast<CachedAllocator*>(GetCpuAllocator())) {
    allocator->finalize();
    printStatistics("CPU", allocator);
  }
  if (auto* allocator = dynamic_cast<CachedAllocator*>(GetCudaAllocator())) {
    allocator->finalize();
    printStatistics("CUDA", allocator);
  }
}

void AllocatorBackend::printStatistics(const std::string& title, CachedAllocator* allocator) const {
  const size_t count = allocator->getRequestCount();
  const size_t bytes_requested = allocator->getTotalBytesRequested();
  const size_t bytes_allocated = allocator->getTotalBytesAllocated();
  const size_t bytes_deallocator = allocator->getTotalBytesDeallocated();
  const double reuse =
      1.0 - static_cast<double>(bytes_allocated)
            / static_cast<double>(bytes_requested == 0 ? 1 : bytes_requested);
  const double duration = ToSeconds(NowCount() - start_time_);

  std::vector<char> buffer(10 * 1024);
  const size_t used_size = std::snprintf(buffer.data(), buffer.size(),
      "|====================================================|\n"
      "|        Big Data Memory Allocation Statistics       |\n"
      "|        %10s                                  |\n"
      "|====================================================|\n"
      "| Duration            |           %16.1f s |\n"
      "| Rate                |               %10.1f 1/s |\n"
      "| Count (Total)       |             %16zd |\n"
      "| Request Rate        |              %10.1f MB/s |\n"
      "| Requested (Total)   |          %16zd MB |\n"
      "| Allocated (Total)   |          %16zd MB |\n"
      "| Deallocated (Total) |          %16zd MB |\n"
      "| Potentially Lost    |           %16zd B |\n"
      "| Efficiency          |                      %5.1f %% |\n"
      "|====================================================|\n",
      title.c_str(),
      duration,
      static_cast<double>(count) / duration,
      count,
      static_cast<double>(bytes_requested) / 1048576.0 / duration,
      bytes_requested / 1048576,
      bytes_allocated / 1048576,
      bytes_deallocator / 1048576,
      bytes_allocated - bytes_deallocator,
      100.0 * reuse);
  LOG_INFO("\n%s", std::string(buffer.data(), used_size).c_str());
}

}  // namespace alice
}  // namespace isaac
