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
#include <set>
#include <string>
#include <vector>

#include "engine/alice/component.hpp"
#include "engine/alice/components/TimeSynchronizer.hpp"

namespace isaac {
namespace alice {

class AsioBackend;

// Receifves messages from a TCP network socket. This components connects to a socket and will
// publish all messages it receives on the socket.
class TcpSubscriber : public Component {
 public:
  TcpSubscriber();
  ~TcpSubscriber();
  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // The IP adress of the remote host from which messages will be received.
  ISAAC_PARAM(std::string, host);
  // The TCP port number on which the remove host is publishing messages.
  ISAAC_PARAM(int, port);
  // If a connection to the remote can not be established or breaks we try to restablish the
  // connection at this interval (in seconds).
  ISAAC_PARAM(double, reconnect_interval, 0.5);
  // If set to true publish timestamp will be set when the message is received; otherwise the
  // original publish timestamp issued by the remote will be used.
  ISAAC_PARAM(bool, update_pubtime, true);

 private:
  // Tries to connect to the remote
  void connectToRemote();
  // Waits for data from the remote
  void receiveFromRemote();

  // If not null, messages received over tcp are assumed to be in sync-time, not app-time.
  alice::TimeSynchronizer* time_synchronizer_;

  AsioBackend* backend_;

  std::vector<uint8_t> buffer_;

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::TcpSubscriber)
