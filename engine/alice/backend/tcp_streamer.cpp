/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "tcp_streamer.hpp"

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/message.hpp"
#include "engine/core/byte.hpp"
#include "engine/core/time.hpp"
#include "engine/gems/serialization/blob.hpp"
#include "engine/gems/serialization/header.hpp"
#include "messages/alice.capnp.h"

namespace isaac {
namespace alice {

namespace {

constexpr size_t kPreHeaderSize = sizeof(size_t);

}  // namespace

struct StreamMessageWriter::Impl {
  std::vector<ByteArrayConstView> proto_blobs;
  std::vector<ByteArrayConstView> buffer_blobs;
  serialization::Header header;
  std::vector<byte> header_buffer;
  ConstMessageBasePtr message;
};

StreamMessageWriter::StreamMessageWriter() {
  impl_.reset(new Impl());
}

StreamMessageWriter::~StreamMessageWriter() {
  impl_.reset();
}

void StreamMessageWriter::write(ConstMessageBasePtr message, TimeSynchronizer* time_synchronizer,
                                const std::string& channel,
                                std::function<void(const uint8_t*, size_t)> ostream) {
  impl_->message = message;
  // Get the segments for the proto message
  auto* proto_message = dynamic_cast<const ProtoMessageBase*>(message.get());
  ASSERT(proto_message != nullptr, "Can currently only transmit proto messages over wire!");
  serialization::CapnpArraysToBlobs(proto_message->segments(), impl_->proto_blobs);
  // Get segments for message buffers
  impl_->buffer_blobs.clear();
  for (const auto& buffer : message->buffers) {
    impl_->buffer_blobs.push_back(ByteArrayConstView{
        reinterpret_cast<const byte*>(buffer.host_buffer().begin()), buffer.host_buffer().size()});
  }
  // Creates the message header
  impl_->header.timestamp = message->pubtime;
  impl_->header.uuid = message->uuid;
  impl_->header.tag = channel;
  impl_->header.proto_id = proto_message->proto_id();
  // If there is a TimeSynchronizer, transmit in sync-time. Otherwise use app-time.
  if (time_synchronizer) {
    impl_->header.acqtime = time_synchronizer->appToSyncTime(message->acqtime);
  } else {
    impl_->header.acqtime = message->acqtime;
  }
  impl_->header.format = 0;
  serialization::BlobsToLengths32u(impl_->proto_blobs, impl_->header.segments);
  serialization::BlobsToLengths32u(impl_->buffer_blobs, impl_->header.buffers);
  // write header
  {
    Serialize(impl_->header, impl_->header_buffer);
    const size_t length = impl_->header_buffer.size();
    ostream(reinterpret_cast<const uint8_t*>(&length), kPreHeaderSize);
    ostream(impl_->header_buffer.data(), length);
  }
  // write proto segments
  for (const auto& blob : impl_->proto_blobs) {
    ostream(blob.begin(), blob.size());
  }
  // write buffers
  for (const auto& blob : impl_->buffer_blobs) {
    ostream(blob.begin(), blob.size());
  }
}

class StreamMessageReader::Impl {
 public:
  using BufferPtr = const uint8_t*;

  // reads bytes from the stream until a full message is formed
  void read(BufferPtr& buffer, size_t& buffer_length) {
    if (bytes_read_ < kPreHeaderSize) {
      readPreHeader(buffer, buffer_length);
    }
    if (buffer_length > 0 && bytes_read_ < kPreHeaderSize + header_length_) {
      readHeader(buffer, buffer_length);
    }
    if (buffer_length > 0 && bytes_read_ < kPreHeaderSize + header_length_ + blobs_length_) {
      readBlobs(buffer, buffer_length);
    }
  }

  // Checks if there is a new message available and if so gets it out
  bool popMessage(std::shared_ptr<BufferedProtoMessage>& out_message, std::string& out_channel) {
    if (bytes_read_ == kPreHeaderSize + header_length_ + blobs_length_) {
      // Move Data to message;
      for (auto& buffer : data_) {
        message_->buffers.emplace_back(std::move(buffer));
      }
      data_.clear();
      out_message = message_;
      out_channel = channel_;
      message_.reset();
      bytes_read_ = 0;
      header_length_ = 0;
      blobs_length_ = 0;
      blobs_.clear();
      blob_index_ = 0;
      return true;
    } else {
      return false;
    }
  }

 private:
  // Reads the pre-header of the message which contains the message header length
  void readPreHeader(BufferPtr& buffer, size_t& buffer_length) {
    ASSERT(bytes_read_ < kPreHeaderSize, "logic error");
    const uint64_t bytes_to_take = std::min(kPreHeaderSize - bytes_read_, buffer_length);
    std::copy(buffer, buffer + bytes_to_take, eight_bytes_.data() + bytes_read_);
    bytes_read_ += bytes_to_take;
    buffer += bytes_to_take;
    buffer_length -= bytes_to_take;
    // interpret data if we have enough
    if (bytes_read_ == kPreHeaderSize) {
      header_length_ = *reinterpret_cast<size_t*>(eight_bytes_.data());
      header_buffer_.resize(header_length_);
    }
  }

  // Reads the message header
  void readHeader(BufferPtr& buffer, size_t& buffer_length) {
    ASSERT(bytes_read_ >= kPreHeaderSize, "logic error");
    const uint64_t offset = bytes_read_ - kPreHeaderSize;
    ASSERT(offset < header_length_, "logic error");
    const uint64_t bytes_to_take = std::min(header_length_ - offset, buffer_length);
    std::copy(buffer, buffer + bytes_to_take, header_buffer_.data() + offset);
    bytes_read_ += bytes_to_take;
    buffer += bytes_to_take;
    buffer_length -= bytes_to_take;
    // check if we have the header completed
    if (bytes_read_ == kPreHeaderSize + header_length_) {
      serialization::Header header;
      const bool ok = Deserialize(header_buffer_, header);
      ASSERT(ok, "could not parse header");
      channel_ = header.tag;
      // get segment offsets
      std::vector<size_t> segment_lengths;
      segment_lengths.reserve(header.segments.size());
      size_t total_proto_length = 0;
      for (const size_t length : header.segments) {
        segment_lengths.push_back(length);
        total_proto_length += length;
      }
      // create new message
      message_.reset(new BufferedProtoMessage(std::vector<uint8_t>(total_proto_length), 0,
                                              std::move(segment_lengths)));
      message_->uuid = *header.uuid;
      message_->acqtime = *header.acqtime;
      message_->pubtime = *header.timestamp;
      if (header.proto_id) {
        message_->setProtoId(*header.proto_id);
      }
      // create message buffers,  buffers will be transfered to message after
      // data has been read into them.
      size_t total_buffer_length = 0;
      data_.clear();
      data_.reserve(header.buffers.size());
      for (const size_t length : header.buffers) {
        data_.emplace_back(length);
        total_buffer_length += length;
      }
      // prepare blobs for reading
      blobs_.clear();
      // Link the proto data buffer to the blobs for reading.
      blobs_.push_back({message_->buffer().data(), total_proto_length});
      // Link the message data buffers to blobs for reading.
      for (auto& buffer : data_) {
        blobs_.push_back({buffer.begin(), buffer.size()});
      }
      blobs_length_ = total_proto_length + total_buffer_length;
    }
  }

  // Reads blobs
  void readBlobs(BufferPtr& buffer, size_t& buffer_length) {
    // Read data from incoming stream into one blob after another. Only take data which we actually
    // need.
    while (buffer_length > 0 && bytes_read_ < kPreHeaderSize + header_length_ + blobs_length_) {
      ASSERT(blob_index_ < blobs_.size(), "Logic error");
      auto& blob = blobs_[blob_index_];
      const size_t len = std::min(blob.size(), buffer_length);
      std::copy(buffer, buffer + len, blob.begin());
      buffer += len;
      buffer_length -= len;
      blob = blob.sub(len);
      bytes_read_ += len;
      if (blob.empty()) {
        blob_index_++;
      }
    }
  }

  // Will be filled with the first kPreHeaderSize bytes which will contain the message header lenght
  std::array<uint8_t, kPreHeaderSize> eight_bytes_;
  // Will be filled with the message header
  std::vector<uint8_t> header_buffer_;

  // number of bytes read so far including kPreHeaderSize bytes and message header
  size_t bytes_read_ = 0;
  // the number of message header bytes
  size_t header_length_ = 0;
  // the number of proto and buffer bytes
  size_t blobs_length_ = 0;

  // Buffers to read
  std::vector<ByteArrayView> blobs_;
  std::vector<CpuBuffer> data_;
  size_t blob_index_;

  // the message which is formed
  std::shared_ptr<BufferedProtoMessage> message_;
  // the channel name under which the message was received
  std::string channel_;
};

StreamMessageReader::StreamMessageReader() {
  impl_.reset(new Impl());
}

StreamMessageReader::~StreamMessageReader() {
  impl_.reset();
}

void StreamMessageReader::read(const uint8_t* buffer, size_t buffer_length) {
  // try to read messages from the received buffer
  while (buffer_length > 0) {
    // continue reading messages
    impl_->read(buffer, buffer_length);
    // check if there is a new message available
    std::shared_ptr<BufferedProtoMessage> message;
    std::string channel;
    if (impl_->popMessage(message, channel)) {
      // publishe the message if someone is interested
      if (on_message) {
        on_message(message, channel);
      }
    }
  }
}

}  // namespace alice
}  // namespace isaac
