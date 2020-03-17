/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <memory>

namespace asio { class io_context; }

namespace isaac {
namespace scheduler {

class Scheduler;

}  // namespace scheduler
}  // namespace isaac

namespace isaac {
namespace alice {

// Hold the ASIO object for other backends
class AsioBackend {
 public:
  AsioBackend(scheduler::Scheduler* scheduler);
  ~AsioBackend();

  void start();
  void stop();

  // The ASIO IO service object
  asio::io_context& io_context();

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
  scheduler::Scheduler* scheduler_;
};

}  // namespace alice
}  // namespace isaac
