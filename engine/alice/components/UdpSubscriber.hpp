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

#include "engine/alice/component.hpp"

namespace isaac {
namespace alice {

class AsioBackend;

// @experimental
// Receives messages from a remote on a UDP network socket
class UdpSubscriber : public Component {
 public:
  UdpSubscriber();
  ~UdpSubscriber();
  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // The local port on which we wait for data
  ISAAC_PARAM(int, port);
  // Number of messages to keep around during message reconstruction
  ISAAC_PARAM(int, message_assembly_slot_count, 5);
  // If set to true publish timestamp will be set when the message is received
  ISAAC_PARAM(bool, update_pubtime, true);

 private:
  // Receives data on the socket asynchronously
  void async_receive();

  AsioBackend* backend_;

  // We are using the pimpl idiom to hide asio in the header
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::UdpSubscriber)
