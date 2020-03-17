/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "udp_packetizer.hpp"

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "asio.hpp"  // NOLINT(build/include)
#include "capnp/serialize.h"
#include "engine/alice/message.hpp"
#include "engine/core/time.hpp"
#include "kj/common.h"
#include "messages/alice.capnp.h"
#include "messages/uuid.hpp"

namespace isaac {
namespace alice {

namespace {

// The size of a "word" (should be 8 bytes)
constexpr size_t kWordSize = sizeof(::capnp::word);
static_assert(kWordSize == 8, "Size of a capnp::word should be 8");

// The length of a header in cap'n'proto words
constexpr size_t kPackageHeaderWords = 5;
// Maximum length for the package payload
constexpr size_t kMtuSize = 1400;
static_assert(kMtuSize % sizeof(capnp::word) == 0,
              "kMtuSize must be a multiple of sizeof(capnp::word)");
// Number of payload words in one package
constexpr size_t kPackagePayloadWords = kMtuSize / kWordSize - kPackageHeaderWords;
// Total maximum number of payload bytes per package
constexpr size_t kPackagePayloadBytes = kWordSize * kPackagePayloadWords;
// Default number of messages which can be assembled at the same time
constexpr size_t kDefaultMessageAssemblySlotCount = 8;

// Creates a non-owning asio buffer from a non-owning kj buffer.
// Please can we have a non-owning buffer type in STD?
asio::const_buffer KjToAsio(const kj::ArrayPtr<const capnp::word>& segment) {
  return asio::const_buffer(segment.asBytes().begin(), segment.asBytes().size());
}

// Computes the number of packages from number of bytes
size_t NumPackagesFromNumBytes(size_t num_bytes) {
  size_t num_packages = num_bytes / kPackagePayloadBytes;
  if (num_packages * kPackagePayloadBytes != num_bytes) {
    num_packages++;
  }
  return num_packages;
}

}  // namespace

void OutPackageBuffer::create(const std::string& channel) {
  // First check we have a ProtoMessage
  proto_message_ = dynamic_cast<const ProtoMessageBase*>(message_.get());
  ASSERT(proto_message_ != nullptr, "Can currently only transmit proto messages over wire!");
  auto message_segments = proto_message_->segments();
  // Prepares the package headers
  package_headers_.clear();
  packages_.clear();
  // Creates the message header
  auto header_root = message_header_builder_.initRoot<MessageHeaderProto>();
  header_root.setProto(0);  // TODO fill proto ID
  header_root.setChannel(channel);
  header_root.setAcqtime(message_->acqtime);
  header_root.setPubtime(message_->pubtime);
  // make it big enough
  auto segment_lengths = header_root.initSegmentLengths(message_segments.size());
  for (size_t i = 0; i < segment_lengths.size(); i++) {
    segment_lengths.set(i, 0);
  }
  // Add the one segment from the message header and then the segments from the payload
  // current pieces which still need to be written
  std::vector<kj::ArrayPtr<const capnp::word>> pieces;
  // total number of words of pieces currently accumulated
  size_t pieces_num_words = 0;

  // writes segment offsets
  for (size_t i = 0; i < message_segments.size(); i++) {
    segment_lengths.set(i, kWordSize * message_segments[i].size());
  }

  // Start with adding the message header
  // Check that message header has only one package which is smaller than the maximum allowed size
  const size_t num_header_segments = message_header_builder_.getSegmentsForOutput().size();
  ASSERT(num_header_segments == 1, "Unexpected multiple segments (%zu) for header.",
         num_header_segments);
  const size_t header_payload = message_header_builder_.getSegmentsForOutput()[0].size();
  ASSERT(header_payload < kPackagePayloadWords,
         "Message header too long: actual=%zu, expected less than %zu.",
         header_payload, kPackagePayloadWords);
  pieces.push_back(message_header_builder_.getSegmentsForOutput()[0]);
  pieces_num_words += pieces.front().size();
  // Computes the message length
  message_length_ = message_header_builder_.getSegmentsForOutput()[0].size();
  for (const auto& segment : message_segments) {
    message_length_ += segment.size();
  }
  message_length_ *= kWordSize;  // words to bytes

  // Add payload segments
  for (const auto& segment : message_segments) {
    // form packages out of pieces until we got everything from the segment
    const capnp::word* ptr = segment.begin();
    size_t words_left_in_segment = segment.size();
    while (words_left_in_segment > 0) {
      // the number of words we take for the next piece
      const size_t num_words_for_piece =
          std::min(words_left_in_segment, kPackagePayloadWords - pieces_num_words);
      // add piece
      pieces.push_back(kj::ArrayPtr<const capnp::word>(ptr, num_words_for_piece));
      pieces_num_words += num_words_for_piece;
      ptr += num_words_for_piece;
      words_left_in_segment -= num_words_for_piece;
      ASSERT(pieces_num_words <= kPackagePayloadWords, "Logic error %d", pieces_num_words);
      // create package
      if (pieces_num_words == kPackagePayloadWords) {
        addPackage(pieces);
        pieces.clear();
        pieces_num_words = 0;
      }
    }
  }
  // publish the last package
  if (pieces_num_words > 0) {
    addPackage(pieces);
    pieces.clear();
    pieces_num_words = 0;
  }
}

void OutPackageBuffer::addPackage(const std::vector<kj::ArrayPtr<const capnp::word>>& pieces) {
  // Create the package header
  auto package_header_builder = std::make_unique<::capnp::MallocMessageBuilder>();
  auto package_header = package_header_builder->initRoot<UdpPackageHeaderProto>();
  ToProto(message_->uuid, package_header.initUuid());
  package_header.setMessageLength(message_length_);
  package_header.setPackageIndex(package_headers_.size());
  ASSERT(package_header_builder->getSegmentsForOutput().size() == 1,
         "Expected one segment for package header");
  auto package_header_segment = package_header_builder->getSegmentsForOutput()[0];
  ASSERT(package_header_segment.size() == kPackageHeaderWords,
         "Expected package header to be %zd words long instead of %zd",
         package_header_segment.size(), kPackageHeaderWords);
  // add the package with its header
  Package package;
  package.reserve(1 + pieces.size());
  package.push_back(KjToAsio(package_header_segment));
  for (const auto& piece : pieces) {
    package.push_back(KjToAsio(piece));
  }
  packages_.emplace_back(std::move(package));
  // we need to hold on to the memory
  package_headers_.emplace_back(std::move(package_header_builder));
}

PackageReassembler::PackageReassembler() {
  reconstructed_messages_timeslots_.resize(kDefaultMessageAssemblySlotCount);
}

PackageReassembler::~PackageReassembler() {
  for (const auto& kvp : puzzles_) {
    LOG_WARNING("Incomplete message %s: %zd/%zd packages still missing", kvp.first.c_str(),
                kvp.second->num_packages_missing.load(), kvp.second->packages_received.size());
  }
}

void PackageReassembler::setMessageAssemblySlotCount(int count) {
  std::unique_lock<std::mutex> lock(mutex_);
  reconstructed_messages_timeslots_.resize(count);
}

size_t PackageReassembler::getMaxBufferLength() const {
  return kWordSize*(kPackagePayloadWords + kPackageHeaderWords);
}

void PackageReassembler::addPackage(const std::vector<uint8_t>& buffer) {
  std::unique_lock<std::mutex> lock(mutex_);
  // The current "time slice", i.e. seconds
  const size_t timeslot = static_cast<size_t>(std::round(ToSeconds(NowCount())))
                          % reconstructed_messages_timeslots_.size();
  ASSERT(timeslot < reconstructed_messages_timeslots_.size(), "Logic error");
  // Interpret buffer as words
  ASSERT(buffer.size() % kWordSize == 0, "Logic error");
  const capnp::word* buffer_words_begin = reinterpret_cast<const ::capnp::word*>(buffer.data());
  const capnp::word* buffer_words_end = reinterpret_cast<const ::capnp::word*>(
      buffer.data() + buffer.size());
  // Get the package header
  kj::ArrayPtr<const capnp::word> package_header_segment(buffer_words_begin, kPackageHeaderWords);
  ::capnp::SegmentArrayMessageReader package_header_reader(
      kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>>(&package_header_segment, 1));
  auto package_header = package_header_reader.getRoot<UdpPackageHeaderProto>();
  // Get the payload
  kj::ArrayPtr<const ::capnp::word> payload_segment(
      buffer_words_begin + kPackageHeaderWords, buffer_words_end);
  // Parse the message UUID
  const Uuid uuid = FromProto(package_header.getUuid());
  // Find message puzzle or create a new one if necessary
  MessagePuzzle* puzzle_ptr = nullptr;
  bool is_new_puzzle = false;
  auto it = puzzles_.find(uuid);
  if (it == puzzles_.end()) {
    // Check if message was already reconstructed
    if (reconstructed_messages_.count(uuid) > 0) {
      LOG_WARNING("Received a package %s for a message which was already reconstructed",
                  uuid.c_str());
      return;
    } else {
      // Remember that we reconstructed this message
      reconstructed_messages_.insert(uuid);
      // Remember when we started with the reconstruction
      reconstructed_messages_timeslots_[timeslot].insert(uuid);
    }
    // start a new puzzle
    auto ok = puzzles_.emplace(uuid, std::make_unique<MessagePuzzle>());
    ASSERT(ok.second, "Failed to add message puzzle");
    it = ok.first;
    is_new_puzzle = true;
  }
  puzzle_ptr = it->second.get();
  ASSERT(puzzle_ptr, "Logic error");
  MessagePuzzle& puzzle = *puzzle_ptr;
  // initialize the puzzle if necessary
  if (is_new_puzzle) {
    puzzle.uuid = uuid;
    const size_t num_bytes = package_header.getMessageLength();
    puzzle.buffer.resize(num_bytes);
    const size_t num_packages = NumPackagesFromNumBytes(num_bytes);
    puzzle.packages_received = std::vector<bool>(num_packages, false);
    puzzle.num_packages_missing = num_packages;
  }
  // Mark the package as received
  const size_t package_index = package_header.getPackageIndex();
  ASSERT(package_index < puzzle.packages_received.size(),
         "Package index %zd/%zd ouf of range (this should not be an assert)",
         package_index, puzzle.packages_received.size());
  {
    if (!puzzle.packages_received[package_index]) {
      puzzle.packages_received[package_index] = true;
    } else {
      LOG_WARNING("Duplicated package %d (this should not be an assert)", package_index);
      return;
    }
  }
  // Handle the message header if we received it
  if (package_index == 0) {
    // Yeah, we got the header! Unparse it now
    std::vector<kj::ArrayPtr<const capnp::word>> message_header_segment = {payload_segment};
    ::capnp::SegmentArrayMessageReader message_header_reader(
      kj::ArrayPtr<const kj::ArrayPtr<const capnp::word>>(message_header_segment.data(), 1));
    auto message_header = message_header_reader.getRoot<MessageHeaderProto>();
    puzzle.proto = message_header.getProto();
    puzzle.channel = message_header.getChannel();
    puzzle.acqtime = message_header.getAcqtime();
    puzzle.pubtime = message_header.getPubtime();
    auto segment_lengths = message_header.getSegmentLengths();
    puzzle.segment_lengths.resize(segment_lengths.size());
    size_t total_length = 0;
    for (size_t i = 0; i < segment_lengths.size(); i++) {
      puzzle.segment_lengths[i] = segment_lengths[i];
      total_length += segment_lengths[i];
    }
    puzzle.header_length = puzzle.buffer.size() - total_length;
  }
  // Copy the received data to the correct location
  auto payload_bytes = payload_segment.asBytes();
  ASSERT(package_index * kPackagePayloadBytes + payload_bytes.size() <= puzzle.buffer.size(),
         "Out of bounds: %zd * %zd + %zd !<= %zd",
         package_index, kPackagePayloadBytes, payload_bytes.size(), puzzle.buffer.size());
  std::copy(payload_bytes.begin(), payload_bytes.end(),
            puzzle.buffer.begin() + package_index * kPackagePayloadBytes);
  // mark the package as processed
  puzzle.num_packages_missing--;
  const bool is_complete = puzzle.num_packages_missing == 0;
  // Check if we received all messages and the puzzle is complete
  if (is_complete) {
    if (!on_message) {
      LOG_WARNING("Received a message on channel '%s' but no one is interested in it :(",
                  puzzle.channel.c_str());
    } else {
      // move out of the map (we erase from the map below)
      const std::string channel = puzzle.channel;  // puzzle gets invalidate by the move
      std::shared_ptr<ProtoMessageBase> message = handleCompleteMessage(std::move(it->second));
      // notify all subscribers
      on_message(message, channel);
    }
    // be done with it
    puzzles_.erase(it);
  }
  // clean old reconstructed messages
  const size_t old_timeslot = (timeslot + 1) % reconstructed_messages_timeslots_.size();
  auto& old_uuids = reconstructed_messages_timeslots_[old_timeslot];
  for (const auto& uuid : old_uuids) {
    reconstructed_messages_.erase(uuid);
    auto it = puzzles_.find(uuid);
    if (it != puzzles_.end()) {
      LOG_WARNING("Discarding incomplete message %s: %zd/%zd packages still missing",
                  it->first.c_str(), it->second->num_packages_missing.load(),
                  it->second->packages_received.size());
      puzzles_.erase(it);
    }
  }
  old_uuids.clear();
}

std::shared_ptr<ProtoMessageBase> PackageReassembler::handleCompleteMessage(
    std::unique_ptr<MessagePuzzle> puzzle_uptr) {
  MessagePuzzle& puzzle = *puzzle_uptr;
  // Create a message based on the received buffers. We move over the memory to avoid copies
  auto message = std::make_shared<BufferedProtoMessage>(
      std::move(puzzle.buffer), puzzle.header_length, std::move(puzzle.segment_lengths));
  message->uuid = puzzle.uuid;
  message->acqtime = puzzle.acqtime;
  message->pubtime = puzzle.pubtime;
  return message;
}

}  // namespace alice
}  // namespace isaac
