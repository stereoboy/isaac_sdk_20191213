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
#include <string>

#include "engine/alice/component.hpp"
#include "engine/alice/message.hpp"

namespace isaac {
namespace alice {

class AsioBackend;

// @experimental
// Publishes messages to a remote using a UDP network socket
class UdpPublisher : public Component {
 public:
  UdpPublisher();
  ~UdpPublisher();
  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // Asynchronously sends a message over the wire
  void async_send(ConstMessageBasePtr message, const std::string& channel);

 private:
  // The host name of the remote to which data is sent
  ISAAC_PARAM(std::string, host)
  // The remote port to which data is sent
  ISAAC_PARAM(int, port)

  AsioBackend* backend_;

  // We are using the pimpl idiom to hide asio in the header
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::UdpPublisher)
