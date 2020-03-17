/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "ProtoToJson.hpp"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "messages/proto_registry.hpp"

namespace isaac {
namespace alice {
namespace {

// Name of the message channel for incoming messages
constexpr char kRxTag[] = "proto";

}  // namespace

void ProtoToJson::start() {
  std::unordered_set<std::string> triggers{full_name() + "/" + kRxTag};
  tickOnEvents(triggers);

  ledger_ = node()->getComponent<MessageLedger>();
  ASSERT(ledger_ != nullptr, "MessageLedger is needed");

  json_codec_ = std::make_unique<::capnp::JsonCodec>();
}

void ProtoToJson::tick() {
  int message_count = 0;
  std::vector<alice::ConstMessageBasePtr> msgs;
  ledger_->readAllNew({this, kRxTag},
                      [&msgs, &message_count](const alice::ConstMessageBasePtr& msg) {
                        message_count++;
                        msgs.push_back(msg);
                      });

  // Handles all messages and add them to buffer pool
  for (auto& msg : msgs) {
    // Get the proto message
    const ProtoMessageBase* proto_msg = static_cast<const ProtoMessageBase*>(msg.get());
    if (proto_msg == nullptr) {
      reportFailure("Only proto messages are supported");
      return;
    }

    // Encode proto as JSON string
    ::capnp::ReaderOptions options;
    options.traversalLimitInWords = kj::maxValue;
    ::capnp::SegmentArrayMessageReader reader(proto_msg->segments(), options);
    auto maybe_reader = GetRootReaderByTypeId(proto_msg->proto_id(), reader);
    if (!maybe_reader) {
      reportFailure("Unknown proto ID: %lld", proto_msg->proto_id());
      return;
    }
    const std::string json_str = (::kj::StringPtr)json_codec_->encode(*maybe_reader);

    // Decode string to JSON
    auto json_payload = serialization::ParseJson(json_str);
    if (!json_payload) {
      reportFailure("Could not parse JSON object from string");
      return;
    }

    // Copy buffers
    for (const auto& buffer : proto_msg->buffers) {
      tx_json().buffers().push_back(buffer.clone());
    }

    // Publishes the json message
    tx_json().publish(std::move(*json_payload), proto_msg->acqtime, proto_msg->type);
  }
}

}  // namespace alice
}  // namespace isaac
