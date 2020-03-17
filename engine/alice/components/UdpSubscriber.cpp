/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "UdpSubscriber.hpp"

#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "engine/alice/application.hpp"
#include "engine/alice/backend/asio_backend.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/udp_packetizer.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/message.hpp"
#include "engine/alice/node.hpp"
#include "engine/core/logger.hpp"

namespace isaac {
namespace alice {

struct UdpSubscriber::Impl {
  std::mutex mutex;
  std::unique_ptr<asio::ip::udp::socket> socket;
  asio::ip::udp::endpoint remote_endpoint;
  PackageReassembler package_reassembler;
};

UdpSubscriber::UdpSubscriber() {
  impl_ = std::make_unique<Impl>();
}

UdpSubscriber::~UdpSubscriber() {
  ASSERT(impl_->socket == nullptr, "Socket not disconnected");
  impl_.reset();
}

void UdpSubscriber::initialize() {
  backend_ = node()->app()->backend()->asio_backend();
}

void UdpSubscriber::start() {
  const int port = get_port();
  impl_->package_reassembler.setMessageAssemblySlotCount(get_message_assembly_slot_count());
  // forward received messages to the message ledger
  auto* message_ledger = node()->getComponent<MessageLedger>();
  impl_->package_reassembler.on_message = [this, message_ledger](ProtoMessageBasePtr message,
                                                                 const std::string& channel) {
    if (get_update_pubtime()) {
      // Change the publish time to the time when the message was published in this
      // application.
      message->pubtime = node()->clock()->timestamp();
    }
    message_ledger->provide({this, channel}, message);
  };
  // start the socket
  ASSERT(impl_->socket == nullptr, "UDP subscriber already started");
  LOG_INFO("Starting UDP subscriber on port %d", port);
  impl_->socket = std::make_unique<asio::ip::udp::socket>(
      backend_->io_context(), asio::ip::udp::endpoint(asio::ip::udp::v4(), port));
  async_receive();
}

void UdpSubscriber::stop() {
  // close the socket
  LOG_INFO("Stopping UDP subscriber on port %d", get_port());
  // FIXME host/port in config might have changed
  std::lock_guard<std::mutex> lock(impl_->mutex);
  impl_->socket.reset();
}

void UdpSubscriber::deinitialize() {
  backend_ = nullptr;
}

void UdpSubscriber::async_receive() {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  const size_t count = impl_->package_reassembler.getMaxBufferLength();
  auto buffer = std::make_shared<std::vector<uint8_t>>(count);
  impl_->socket->async_receive_from(
      asio::buffer(buffer->data(), buffer->size()), impl_->remote_endpoint,
      [this, buffer](const asio::error_code& ec, size_t bytes_read) {
        if (ec) {
          if (ec == asio::error::operation_aborted) {
            return;
          }
          LOG_ERROR("Socket failed to receive data: %s", ec.message().c_str());
        } else {
          buffer->resize(bytes_read);
          impl_->package_reassembler.addPackage(*buffer);
        }
        async_receive();
      });
}

}  // namespace alice
}  // namespace isaac
