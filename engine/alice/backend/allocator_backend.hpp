/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cstdint>
#include <string>

#include "engine/core/allocator/cached_allocator.hpp"

namespace isaac {
namespace alice {

class Application;

// Manages memory allocator
class AllocatorBackend  {
 public:
  AllocatorBackend(Application* app);

  void start();
  void stop();

 private:
  // Prints allocation statistics to the console.
  void printStatistics(const std::string& title, CachedAllocator* allocator) const;

  Application* app_;
  int64_t start_time_;
};

}  // namespace alice
}  // namespace isaac
