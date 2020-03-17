/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

namespace isaac {
namespace alice {

// Indicates the current lifecycle stage of an object, e.g. a node or component
enum class Lifecycle {
  // The object was never started
  kNeverStarted = 0,
  // The object began to start but has not yet finished to start
  kBeforeStart = 10,
  // The object has finished starting and is ready
  kAfterStart = 11,
  // The object has ticked at least once since start was called
  kRunning = 20,
  // The object began to stop but has not yet finished to stop
  kBeforeStop = 30,
  // The object has stopped
  kAfterStop = 31
};

}  // namespace alice
}  // namespace isaac
