/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <atomic>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "capnp/common.h"
#include "capnp/serialize.h"
#include "engine/alice/message.hpp"
#include "engine/gems/uuid/uuid.hpp"
#include "kj/common.h"
#include "messages/alice.capnp.h"

namespace isaac {
namespace alice {

// An object which prepares a message for wire transport and holds auxilary buffers
class OutPackageBuffer {
 public:
  // A list of packages to send. Each package consists of two buffers: A header which will allow
  // us to reconstruct the message on the receiving size and the actual contents of the message.
  // The two buffers must be sent as one atomic package, e.g. using writev or send_msg.
  using Package = std::vector<asio::const_buffer>;

  OutPackageBuffer(ConstMessageBasePtr message, const std::string& channel)
  : message_(message) {
    create(channel);
  }

  // disallow copy
  OutPackageBuffer() = delete;
  OutPackageBuffer(const OutPackageBuffer&) = delete;
  OutPackageBuffer& operator=(const OutPackageBuffer&) = delete;

  // A list of atomic packages we want to send over the wire.
  const std::vector<Package>& packages() const { return packages_; }

 private:
  // Runs message packetization
  void create(const std::string& channel);
  // Adds a package
  void addPackage(const std::vector<kj::ArrayPtr<const capnp::word>>& pieces);

  ConstMessageBasePtr message_;
  const ProtoMessageBase* proto_message_;
  ::capnp::MallocMessageBuilder message_header_builder_;
  std::vector<std::unique_ptr<::capnp::MallocMessageBuilder>> package_headers_;
  size_t message_length_;
  std::vector<Package> packages_;
};

// Gets packages received over the wire and re-assembles them into messages
class PackageReassembler {
 public:
  PackageReassembler();
  ~PackageReassembler();

  // Sets the number of messages which can be assembled from packages at the same time
  void setMessageAssemblySlotCount(int count);

  // The maximum length of a package received on the socket
  size_t getMaxBufferLength() const;
  // A buffer with data we read from the socket
  void addPackage(const std::vector<uint8_t>& buffer);

  // This function is called when a new message is ready
  std::function<void(ProtoMessageBasePtr, const std::string&)> on_message;

 private:
  // Used to reassemble a message from individual messages
  struct MessagePuzzle {
    // disallow copy
    MessagePuzzle() {}
    MessagePuzzle& operator=(const MessagePuzzle&) = delete;
    MessagePuzzle(const MessagePuzzle&) = delete;

    // the message buffer containing the header + payload data
    std::vector<uint8_t> buffer;

    // a bitmap to signal which packages we already received
    std::vector<bool> packages_received;

    // the number of packages still missing
    std::atomic<size_t> num_packages_missing;

    // segment lenghts
    std::vector<size_t> segment_lengths;

    // the length of the header
    size_t header_length;

    // header data
    Uuid uuid;
    uint64_t proto = 0;
    std::string channel;
    int64_t acqtime = 0;
    int64_t pubtime = 0;
  };

  // Creates a message from a complete puzzle
  ProtoMessageBasePtr handleCompleteMessage(std::unique_ptr<MessagePuzzle> puzzle);

  std::map<Uuid, std::unique_ptr<MessagePuzzle>> puzzles_;

  // List of already completed messages // FIXME limit in time
  std::unordered_set<Uuid> reconstructed_messages_;
  // Remembers when we received messages so that we can keep the list clean. Timestamps are
  // with 1 second accuracy.
  std::vector<std::set<Uuid>> reconstructed_messages_timeslots_;

  std::mutex mutex_;
};

}  // namespace alice
}  // namespace isaac
