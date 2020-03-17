/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "TcpSubscriber.hpp"

#include <atomic>
#include <memory>
#include <string>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "engine/alice/application.hpp"
#include "engine/alice/backend/asio_backend.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/tcp_streamer.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/node.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/scheduler/scheduler.hpp"

namespace isaac {
namespace alice {

namespace {

constexpr size_t kMaxLength = 1024 * 16;

}  // namespace

struct TcpSubscriber::Impl {
  std::optional<scheduler::JobHandle> job_handle;
  bool stop_requested_;
  bool try_connecting_;
  std::unique_ptr<asio::ip::tcp::socket> socket;
  StreamMessageReader message_reader;
};

TcpSubscriber::TcpSubscriber() {
  impl_ = std::make_unique<Impl>();
  buffer_.resize(kMaxLength);
}

TcpSubscriber::~TcpSubscriber() {
  ASSERT(impl_->socket == nullptr, "Socket not disconnected");
  impl_.reset();
}

void TcpSubscriber::initialize() {
  backend_ = node()->app()->backend()->asio_backend();
}

void TcpSubscriber::start() {
  // Check if time sync is requested
  time_synchronizer_ = node()->getComponentOrNull<alice::TimeSynchronizer>();
  // forward received messages to the message ledger
  auto* message_ledger = node()->getComponent<MessageLedger>();
  impl_->message_reader.on_message = [this, message_ledger](MessageBasePtr message,
                                                            const std::string& channel) {
    if (time_synchronizer_) {
      // Convert acqtime from sync-time to app-time
      message->acqtime = time_synchronizer_->syncToAppTime(message->acqtime);
    }
    if (get_update_pubtime()) {
      // Change the publish time to the time when the message was published in this
      // application.
      message->pubtime = node()->clock()->timestamp();
    }
    message_ledger->provide({this, channel}, message);
  };
  // connect to remote to receive messages
  impl_->stop_requested_ = false;
  connectToRemote();
}

void TcpSubscriber::stop() {
  // close the socket
  LOG_INFO("Stopping TCP receiver for (%s:%d)...", get_host().c_str(), get_port());
  // FIXME host/port in config might have changed
  impl_->stop_requested_ = true;

  if (impl_->job_handle) {
    node()->app()->backend()->scheduler()->destroyJobAndWait(*(impl_->job_handle));
  }
  impl_->job_handle = std::nullopt;

  if (impl_->socket) {
    std::error_code ec;
    impl_->socket->close(ec);
    if (ec) {
      LOG_ERROR("Failed to close socket: %s", ec.message().c_str());
    }
    impl_->socket.reset();
  }
  LOG_INFO("Stopping TCP receiver for (%s:%d)... DONE", get_host().c_str(), get_port());
}

void TcpSubscriber::deinitialize() {
  backend_ = nullptr;
}

void TcpSubscriber::connectToRemote() {
  impl_->try_connecting_ = false;

  if (impl_->job_handle) {
    node()->app()->backend()->scheduler()->destroyJobAndWait(*(impl_->job_handle));
  }
  impl_->job_handle = std::nullopt;

  if (impl_->stop_requested_) {
    return;
  }
  impl_->try_connecting_ = true;
  // TODO Don't spin a thread here
  // ToDo: If there is not a default blocker group this will fail.
  scheduler::JobDescriptor job_descriptor;
  job_descriptor.priority = 0;
  job_descriptor.execution_mode = scheduler::ExecutionMode::kBlockingOneShot;
  job_descriptor.name = "connectToRemote";
  job_descriptor.action = [this] {
    // close the socket first
    if (impl_->socket) {
      std::error_code ec;
      impl_->socket->close(ec);
      impl_->socket.reset();
    }
    // resolve host
    const std::string host = get_host();
    const int port = get_port();
    LOG_INFO("TCP receiver trying to connect to remote (%s:%d)...", host.c_str(), port);
    asio::ip::tcp::resolver resolver(backend_->io_context());
    std::error_code ec;
    asio::ip::tcp::resolver::query query(host, std::to_string(port));
    auto it = resolver.resolve(query, ec);
    if (ec) {
      LOG_ERROR("Could not resolve host %s (port %d): %s", host.c_str(), port,
                ec.message().c_str());
      return;
    }
    // create socket
    impl_->socket = std::make_unique<asio::ip::tcp::socket>(backend_->io_context());
    // try to establish connection until success (or stop requested)
    while (impl_->try_connecting_ && !impl_->stop_requested_) {
      // try to connect
      asio::error_code ec;
      impl_->socket->connect(*it, ec);
      // check if successful
      if (ec) {
        if (ec == asio::error::operation_aborted) {
          return;
        }
        // try again
        const double reconnect_interval = get_reconnect_interval();
        LOG_WARNING("Failed to connect to remote. Will try again in %f seconds.",
                    reconnect_interval);
        Sleep(SecondsToNano(reconnect_interval));  // TODO make this interruptible
      } else {
        // start receiving data
        LOG_INFO("Successfully connected to remote (%s:%d). Will start receiving.", host.c_str(),
                 port);
        receiveFromRemote();
        return;
      }
    }
  };
  impl_->job_handle = node()->app()->backend()->scheduler()->createJobAndStart(job_descriptor);
  ASSERT(impl_->job_handle, "Unable to start TCPSubscriber connection");
}

void TcpSubscriber::receiveFromRemote() {
  if (impl_->stop_requested_) return;
  impl_->socket->async_receive(
      asio::buffer(buffer_.data(), kMaxLength),
      [this](const asio::error_code& ec, size_t bytes_read) {
        if (ec) {
          // If we receive an error we print a message and try to reconnect.
          if (ec == asio::error::eof) {
            LOG_WARNING(
                "EOF encountered which likely indicates that the connection was closed by "
                "the remote. Will try to reconnect.");
          } else {
            LOG_ERROR("Socket failed to receive data. Will try to reconnect.");
          }
          connectToRemote();
        } else {
          // We have successfully read data, will forward it to the message re-assembler and then
          // try to receive more.
          impl_->message_reader.read(buffer_.data(), bytes_read);
          receiveFromRemote();
        }
      });
}

}  // namespace alice
}  // namespace isaac
