/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "UdpPublisher.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "engine/alice/application.hpp"
#include "engine/alice/backend/asio_backend.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/udp_packetizer.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/node.hpp"
#include "engine/core/assert.hpp"
#include "engine/core/logger.hpp"

namespace isaac {
namespace alice {

struct UdpPublisher::Impl {
  std::mutex mutex;
  std::unique_ptr<asio::ip::udp::socket> socket;
  asio::ip::udp::endpoint remote_endpoint;
};

UdpPublisher::UdpPublisher() {
  impl_ = std::make_unique<Impl>();
}

UdpPublisher::~UdpPublisher() {
  ASSERT(impl_->socket == nullptr, "Socket not disconnected");
  impl_.reset();
}

void UdpPublisher::initialize() {
  backend_ = node()->app()->backend()->asio_backend();
  // Send all messages arriving in the message ledger which are meant for the UDP publisher
  MessageLedger* ledger = node()->getComponent<MessageLedger>();
  ASSERT(ledger, "UdpPublisher requires MessageLedger component");
  ledger->addOnConnectAsRxCallback(
      [ledger, this](const MessageLedger::Endpoint& tx, const MessageLedger::Endpoint& rx) {
        if (rx.component != this) {
          return;
        }
        ledger->addOnMessageCallback(rx, tx.component, [this, rx](ConstMessageBasePtr message) {
          this->async_send(message, rx.tag);
        });
      });
}

void UdpPublisher::start() {
  const std::string host = get_host();
  const int port = get_port();
  // Initiate UDP connection to remote
  ASSERT(impl_->socket == nullptr, "UDP transmitter already started");
  LOG_INFO("Starting UDP transmitter which connects to %s:%d", host.c_str(), port);
  asio::ip::udp::resolver resolver(backend_->io_context());
  asio::ip::udp::resolver::query query(asio::ip::udp::v4(), host, std::to_string(port));
  impl_->remote_endpoint = *resolver.resolve(query);
  impl_->socket = std::make_unique<asio::ip::udp::socket>(backend_->io_context());
  impl_->socket->open(asio::ip::udp::v4());
}

void UdpPublisher::stop() {
  // stop forwarding messages
  node()->getComponent<MessageLedger>()->removeCustomer(this);
  // close UDP socket
  LOG_INFO("Stopping UDP transmitter for (%s:%d)...", get_host().c_str(), get_port());
  // FIXME host/port in config might have changed
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->socket.reset();
}

void UdpPublisher::deinitialize() {
  backend_ = nullptr;
}

void UdpPublisher::async_send(ConstMessageBasePtr message, const std::string& channel) {
  ASSERT(message->buffers.empty(), "buffers are not supported over UDP");
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->socket) {
    LOG_WARNING("Could not send message because socket is not yet started");
    return;
  }
  auto buffer = std::make_shared<OutPackageBuffer>(message, channel);
  for (auto& package : buffer->packages()) {
    impl_->socket->async_send_to(
        package, impl_->remote_endpoint, [this, buffer](std::error_code ec, size_t num_sent) {
          if (ec) {
            LOG_ERROR("Could not write to socket: %s", ec.message().c_str());
          }
        });
  }
}

}  // namespace alice
}  // namespace isaac
