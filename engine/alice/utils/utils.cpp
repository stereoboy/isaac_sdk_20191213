/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "utils.hpp"

#include <algorithm>
#include <memory>
#include <utility>
#include <vector>

#include "engine/gems/serialization/blob.hpp"
#include "engine/gems/serialization/header.hpp"

namespace isaac {
namespace alice {

void WriteMessageToCask(ConstMessageBasePtr message, cask::Cask& cask) {
  const ProtoMessageBase* proto_message = dynamic_cast<const ProtoMessageBase*>(message.get());
  ASSERT(proto_message != nullptr, "Only proto messages are supported for logging");

  // Prepares message proto
  std::vector<ByteArrayConstView> segments;
  serialization::CapnpArraysToBlobs(proto_message->segments(), segments);

  // Prepares buffers
  std::vector<ByteArrayConstView> buffer_blobs;
  for (const auto& buffer : message->buffers) {
    buffer_blobs.push_back(
        ByteArrayConstView{buffer.host_buffer().begin(), buffer.host_buffer().size()});
  }

  // Prepares message header
  serialization::Header header;
  header.proto_id = proto_message->proto_id();
  header.timestamp = message->pubtime;
  header.acqtime = message->acqtime;
  serialization::BlobsToLengths32u(segments, header.segments);
  serialization::BlobsToLengths32u(buffer_blobs, header.buffers);

  // Prepares message header length
  size_t header_length;
  if (!Size(header, true, &header_length, nullptr)) {
    PANIC("could not compute header size");
  }

  // Prepares total segment length
  const size_t segment_length = serialization::AccumulateLength(segments);
  const size_t buffer_length = serialization::AccumulateLength(buffer_blobs);

  // Writes data
  cask.keyValueWrite(message->uuid, header_length + segment_length + buffer_length,
                     [&](byte* begin, byte* end) {
                       begin = Serialize(header, begin, end);
                       ASSERT(begin, "error serializing header");
                       begin = serialization::CopyAll(segments, begin, end);
                       begin = serialization::CopyAll(buffer_blobs, begin, end);
                     });
}

MessageBasePtr ReadMessageFromCask(const Uuid& message_uuid, cask::Cask& cask) {
  serialization::Header header;
  std::vector<byte> proto_buffer;
  std::vector<SharedBuffer> buffers;
  cask.keyValueRead(message_uuid, [&](const byte* begin, const byte* end) {
    begin = Deserialize(begin, end, header);
    // compute proto buffer length
    const size_t total_proto_length =
        std::accumulate(header.segments.begin(), header.segments.end(), size_t{0});
    // copy out proto buffer
    proto_buffer.resize(total_proto_length);
    std::copy(begin, begin + total_proto_length, proto_buffer.begin());
    begin += total_proto_length;
    // de-serialize buffers
    buffers.reserve(header.buffers.size());
    for (const size_t length : header.buffers) {
      CpuBuffer output(length);
      std::copy(begin, begin + length, output.begin());
      buffers.emplace_back(std::move(output));
      begin += length;
    }
  });
  ASSERT(header.timestamp, "Timestamp missing in message header");
  ASSERT(header.acqtime, "Acqtime timestamp missing in message header");
  ASSERT(!header.segments.empty(), "Message must have at least one segment");
  std::vector<size_t> segment_offsets(header.segments.size());
  for (size_t i = 0; i < segment_offsets.size(); i++) {
    segment_offsets[i] = static_cast<size_t>(header.segments[i]);
  }

  auto message = std::make_shared<BufferedProtoMessage>(std::move(proto_buffer), 0,
                                                        std::move(segment_offsets));
  message->buffers = std::move(buffers);
  message->uuid = message_uuid;
  message->pubtime = *header.timestamp;
  message->acqtime = *header.acqtime;
  if (header.proto_id) {
    message->setProtoId(*header.proto_id);
  }
  return message;
}

}  // namespace alice
}  // namespace isaac
