/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "TcpPublisher.hpp"

#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "engine/alice/application.hpp"
#include "engine/alice/backend/asio_backend.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/tcp_streamer.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/message.hpp"
#include "engine/alice/node.hpp"
#include "engine/core/logger.hpp"

namespace isaac {
namespace alice {

struct TcpPublisher::Impl {
  std::unique_ptr<asio::ip::tcp::acceptor> acceptor;
  // Current socket looking for a connection
  std::unique_ptr<asio::ip::tcp::socket> pending_socket;
  // List of connected sockets
  // Note: Unfortunately async_accept does not support non-copyable lambdas, and thus we can not
  // use unique pointers here.
  std::vector<std::unique_ptr<asio::ip::tcp::socket>> sockets;
  std::mutex sockets_mutex;

  StreamMessageWriter stream_message_writer;
};

TcpPublisher::TcpPublisher() {
  impl_ = std::make_unique<Impl>();
}

TcpPublisher::~TcpPublisher() {
  ASSERT(impl_->sockets.empty(), "Sockets not disconnected");
  impl_.reset();
}

void TcpPublisher::initialize() {
  // Initialize time_synchronizer_ pointer to null here because the send() function below, which
  // uses time_synchronizer_ may be called before TcpPublisher::start(), where we set
  // time_synchronizer_. We don't move the getComponentOrNull() call in TcpPublisher::start() to
  // here because it is too soon to check that here: TimeSynchronizer component may or may not be
  // added to the node at this point.
  time_synchronizer_ = nullptr;
  backend_ = node()->app()->backend()->asio_backend();
  // Send all messages arriving in the message ledger which are meant for the UDP publisher
  node()->getComponent<MessageLedger>()->addOnMessageCallback(
      [this](const MessageLedger::Endpoint& channel, const ConstMessageBasePtr& message) {
        if (channel.component != this) return;
        this->send(message, channel.tag);
      });
}

void TcpPublisher::start() {
  // Check if time sync is requested
  time_synchronizer_ = node()->getComponentOrNull<alice::TimeSynchronizer>();
  // open TCP port and wait for connections
  const int port = get_port();
  ASSERT(impl_->acceptor == nullptr, "Server already started");
  LOG_INFO("Starting TCP publisher on port %d", port);
  impl_->acceptor = std::make_unique<asio::ip::tcp::acceptor>(backend_->io_context());
  asio::ip::tcp::endpoint endpoint(asio::ip::tcp::v4(), port);
  std::error_code ec;
  impl_->acceptor->open(endpoint.protocol(), ec);
  if (ec) {
    reportFailure("Failed to open TCP acceptor for V4 protocol. Will not publish any data.");
    return;
  }
  impl_->acceptor->set_option(asio::socket_base::reuse_address(true), ec);
  if (ec) {
    reportFailure("Failed to enable socket reuse for port %d. Will not publish any data.", port);
    return;
  }
  impl_->acceptor->bind(endpoint, ec);
  if (ec) {
    reportFailure(
        "Failed to bind TCP endpoint at port %d. Is the port already in use? "
        "Will not publish any data.",
        port);
    return;
  }
  impl_->acceptor->listen(asio::socket_base::max_connections, ec);
  if (ec) {
    reportFailure("Failed to listen on TCP endpoint. Will not publish any data.");
    return;
  }
  acceptConnection();
}

void TcpPublisher::stop() {
  // stop forwarding messages
  node()->getComponent<MessageLedger>()->removeCustomer(this);
  // close connection
  // FIXME we might not print the port we opened
  LOG_INFO("Stopping TCP publisher on port %d", get_port());
  std::lock_guard<std::mutex> lock(impl_->sockets_mutex);
  if (impl_->acceptor) {
    impl_->acceptor->cancel();
    impl_->acceptor.reset();
  }
  impl_->pending_socket.reset();
  for (auto& p : impl_->sockets) {
    if (p->is_open()) {
      p->shutdown(asio::ip::tcp::socket::shutdown_both);
      p->close();
    }
  }
  impl_->sockets.clear();
}

void TcpPublisher::deinitialize() {
  backend_ = nullptr;
}

void TcpPublisher::acceptConnection() {
  ASSERT(impl_->acceptor, "Acceptor not initialized");
  impl_->pending_socket = std::make_unique<asio::ip::tcp::socket>(backend_->io_context());
  ASSERT(impl_->pending_socket, "Could not create socket");
  impl_->acceptor->async_accept(*impl_->pending_socket, [this](std::error_code ec) {
    if (ec) {
      if (ec == asio::error::operation_aborted) {
        return;
      }
      LOG_WARNING("Failed to accept connection. Will try again.");
    } else {
      LOG_INFO("TCP publisher successfully accepted conntection");
      std::lock_guard<std::mutex> lock(impl_->sockets_mutex);
      impl_->sockets.emplace_back(std::move(impl_->pending_socket));
      impl_->pending_socket = nullptr;
    }
    acceptConnection();
  });
}

void TcpPublisher::send(ConstMessageBasePtr message, const std::string& channel) {
  std::lock_guard<std::mutex> lock(impl_->sockets_mutex);
  // memorize invalid sockets so that we can close them
  std::set<asio::ip::tcp::socket*> invalid_sockets;
  // stream the message
  impl_->stream_message_writer.write(
      message, time_synchronizer_, channel,
      [this, &invalid_sockets](const uint8_t* buffer, size_t count) {
        // send message data pieces to all connected sockets
        for (auto& socket : impl_->sockets) {
          asio::ip::tcp::socket* socket_ptr = socket.get();
          if (invalid_sockets.count(socket_ptr) > 0) {
            continue;
          }
          std::error_code ec;
          const size_t num_sent = socket_ptr->send(
              std::array<asio::const_buffer, 1>{asio::const_buffer(buffer, count)}, 0, ec);
          bool has_error = false;
          if (ec) {
            LOG_WARNING("Could not write to socket");
            has_error = true;
          }
          if (num_sent != count) {
            LOG_WARNING("Did not sent all bytes (%zd instead of %zd)", num_sent, count);
            has_error = true;
          }
          // close connection
          if (has_error) {
            invalid_sockets.insert(socket_ptr);
          }
        }
      });
  // remove invalid sockets
  if (!invalid_sockets.empty()) {
    LOG_WARNING("Closed %zd invalid connection(s)", invalid_sockets.size());
    for (auto* socket : invalid_sockets) {
      // TODO Do we need to close / shutdown / cancel the socket?
      impl_->sockets.erase(std::find_if(impl_->sockets.begin(), impl_->sockets.end(),
                                        [socket](const auto& x) { return x.get() == socket; }));
    }
  }
}

}  // namespace alice
}  // namespace isaac
