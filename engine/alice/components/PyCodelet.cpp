/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "PyCodelet.hpp"

#include <algorithm>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/component.hpp"
#include "engine/alice/message.hpp"
#include "engine/gems/serialization/capnp.hpp"
#include "engine/gems/serialization/json.hpp"

namespace isaac {
namespace alice {

using byte = uint8_t;

void PyCodelet::start() {
  pycodelet_flow_control_.cppDelegateJob("py_start");
}

void PyCodelet::tick() {
  pycodelet_flow_control_.cppDelegateJob("py_tick");
}

void PyCodelet::stop() {
  pycodelet_flow_control_.stop();
}

void PyCodelet::addRxHook(const std::string& rx_hook) {
  addRxMessageHook(rx_hook);
}
void PyCodelet::addTxHook(const std::string& tx_hook) {
  addTxMessageHook(tx_hook);
}

void PyCodelet::synchronizeWithTags(const std::string& tag1, const std::string& tag2) {
  synchronize(*getRxMessageHook(tag1), *getRxMessageHook(tag2));
}

void PyCodelet::tickOnMessageWithTag(const std::string& tag) {
  tickOnMessage(*getRxMessageHook(tag));
}

void PyCodelet::getRxBufferWithTagAndIndex(const std::string& tag, const int idx,
                                           std::string& bytes) {
  const auto rx = getRxMessageHook(tag);
  const auto& buffers = rx->buffers();
  ASSERT(static_cast<int>(buffers.size()) > idx, "invalid buffer index");

  bytes.resize(buffers[idx].host_buffer().size());
  std::copy(buffers[idx].host_buffer().begin(), buffers[idx].host_buffer().end(), bytes.begin());
}

size_t PyCodelet::addTxBuffer(const std::string& tag, size_t size, void* p) {
  auto tx = getTxMessageHook(tag);
  CpuBuffer buffer(size);
  std::copy(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + size, buffer.begin());
  tx->buffers().emplace_back(std::move(buffer));
  return tx->buffers().size();
}

const BufferedProtoMessage* PyCodelet::publish(const std::string& tag, const std::string& bytes,
                                               std::optional<int64_t> acqtime) {
  // TODO - replace this with pycapnp to_segments()/from_segments() API
  // read the number of segments from the string message
  std::vector<uint8_t> buffer;
  std::vector<size_t> segment_lengths;
  isaac::serialization::StringToCapnpBuffer(bytes, segment_lengths, buffer);

  // prepare the message and publish it through the tx message hook
  auto message = std::make_shared<BufferedProtoMessage>(buffer, 0, segment_lengths);
  publishMessage(tag, message, acqtime);
  return message.get();
}

void PyCodelet::receive(const std::string& tag, std::string& bytes) {
  // get the message from the rx message hook
  const ProtoMessageBase* proto_msg = dynamic_cast<const ProtoMessageBase*>(getMessage(tag));
  ASSERT(proto_msg != nullptr, "Only proto messages are supported for pycodelet");
  isaac::serialization::CapnpSegmentsToString(proto_msg->segments(), bytes);
}

bool PyCodelet::available(const std::string& tag) {
  return isMessageAvailable(tag);
}

int64_t PyCodelet::getRxPubtime(const std::string& tag) {
  return getRxMessageHook(tag)->pubtime();
}

int64_t PyCodelet::getRxAcqtime(const std::string& tag) {
  return getRxMessageHook(tag)->acqtime();
}

const Uuid& PyCodelet::getRxUuid(const std::string& tag) {
  return getRxMessageHook(tag)->message_uuid();
}

std::string PyCodelet::pythonWaitForJob() {
  auto job = pycodelet_flow_control_.pythonWaitForJob();
  if (job) return *job;
  return std::string("");  // we use empty string to represent the stopping signal
}

void PyCodelet::pythonJobFinished() {
  pycodelet_flow_control_.pythonJobFinished();
}

void PyCodelet::show(const std::string& sop_json) {
  node()->sight().show(this, Json::parse(sop_json));
}

}  // namespace alice
}  // namespace isaac
