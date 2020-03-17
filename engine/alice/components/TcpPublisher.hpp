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
#include <mutex>
#include <string>

#include "engine/alice/component.hpp"
#include "engine/alice/components/TimeSynchronizer.hpp"
#include "engine/alice/message.hpp"

namespace isaac {
namespace alice {

class AsioBackend;

// Sends messages via a TCP network socket. This components waits for clients to connect and will
// forward all messages which are sent to it to connected clients.
class TcpPublisher : public Component {
 public:
  TcpPublisher();
  ~TcpPublisher();

  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // The TCP port number used to wait for connections and to publish messages.
  ISAAC_PARAM(int, port);

  // Sends a message to remotes
  void send(ConstMessageBasePtr message, const std::string& channel);

 private:
  // Asynchronously accepts a socket connection
  void acceptConnection();

  // If not null, messages sent over tcp will be in sync-time, not app-time.
  alice::TimeSynchronizer* time_synchronizer_;

  AsioBackend* backend_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::TcpPublisher)
