/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <functional>
#include <memory>
#include <string>

#include "engine/alice/components/TimeSynchronizer.hpp"
#include "engine/alice/message.hpp"

namespace isaac {
namespace alice {

// Writes a proto message to a stream
class StreamMessageWriter {
 public:
  StreamMessageWriter();
  ~StreamMessageWriter();
  // Writes a proto message to a stream
  // Calls ostream one or more times to write data to the stream.
  // If a TimeSynchronizer is provided, message will be written in sync-time, not app-time.
  void write(ConstMessageBasePtr message, TimeSynchronizer* time_synchronizer,
             const std::string& channel, std::function<void(const uint8_t*, size_t)> ostream);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Reads a proto message from a stream
class StreamMessageReader {
 public:
  StreamMessageReader();
  ~StreamMessageReader();
  // Called by read when a complete message was received
  std::function<void(MessageBasePtr, const std::string&)> on_message;
  // Add another block of data for message reconstructions
  void read(const uint8_t* buffer, size_t count);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace alice
}  // namespace isaac
