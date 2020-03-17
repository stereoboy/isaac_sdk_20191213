/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "asio_backend.hpp"

#include <memory>

#include "asio.hpp"  // NOLINT(build/include)
#include "engine/core/assert.hpp"
#include "engine/core/singleton.hpp"
#include "engine/gems/scheduler/scheduler.hpp"

namespace isaac {
namespace alice {

class AsioBackend::Impl {
 public:
  Impl(scheduler::Scheduler* scheduler) : scheduler_(scheduler), io_service_finished_(false) {
    // Create some work so that the io service never stops
    work_ = std::make_unique<asio::io_context::work>(io_service_);
  }
  asio::io_context& io_context() {
    ASSERT(!io_service_finished_, "IO service stopped prematurely");
    return io_service_;
  }
  void run() {
    LOG_INFO("Starting ASIO service");
    // ToDo: If there is not a default blocker group this will fail.
    scheduler::JobDescriptor job_descriptor;
    job_descriptor.priority = 0;
    job_descriptor.execution_mode = scheduler::ExecutionMode::kBlocking;
    job_descriptor.name = "ASIO Service";
    job_descriptor.action = [this] {
      io_service_.run();
      io_service_finished_ = true;
    };

    job_handle_ = scheduler_->createJobAndStart(job_descriptor);
    ASSERT(job_handle_, "Unable to bring up ASIO Backend");
  }
  ~Impl() {
    ASSERT(!io_service_finished_, "IO service stopped prematurely");
    LOG_INFO("Stopping ASIO service");
    work_.reset();
    io_service_.stop();
    scheduler_->destroyJobAndWait(*job_handle_);
  }

 private:
  std::optional<scheduler::JobHandle> job_handle_;
  std::unique_ptr<asio::io_context::work> work_;
  asio::io_context io_service_;
  scheduler::Scheduler* scheduler_;
  bool io_service_finished_;
};

AsioBackend::~AsioBackend() {
  // for unique_ptr destructor
}

AsioBackend::AsioBackend(scheduler::Scheduler* scheduler) : scheduler_(scheduler) {
  // for unique_ptr destructor
}

void AsioBackend::start() {
  impl_.reset(new Impl(scheduler_));
  impl_->run();
}

void AsioBackend::stop() {
  impl_.reset();
}

asio::io_context& AsioBackend::io_context() {
  return impl_->io_context();
}

}  // namespace alice
}  // namespace isaac
