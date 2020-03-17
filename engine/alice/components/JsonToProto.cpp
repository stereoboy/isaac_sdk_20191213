/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "JsonToProto.hpp"

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "engine/gems/serialization/json.hpp"
#include "messages/proto_registry.hpp"

namespace isaac {
namespace alice {

namespace {

// Name of the message channel for incoming messages
constexpr char kRxTag[] = "json";

}  // namespace

void JsonToProto::start() {
  std::unordered_set<std::string> triggers{full_name() + "/" + kRxTag};
  tickOnEvents(triggers);

  ledger_ = node()->getComponent<MessageLedger>();
  ASSERT(ledger_ != nullptr, "MessageLedger is needed");

  json_codec_ = std::make_unique<::capnp::JsonCodec>();
}

void JsonToProto::tick() {
  int message_count = 0;
  std::vector<alice::ConstMessageBasePtr> msgs;
  ledger_->readAllNew({this, kRxTag},
                      [&msgs, &message_count](const alice::ConstMessageBasePtr& msg) {
                        message_count++;
                        msgs.push_back(msg);
                      });

  // Handles all messages
  for (auto& msg : msgs) {
    const JsonMessage* json_msg = static_cast<const JsonMessage*>(msg.get());
    if (json_msg == nullptr) {
      reportFailure("Only JsonMessage is supported");
      return;
    }

    // Creates proto message and sends it out.
    auto message_builder = std::make_unique<::capnp::MallocMessageBuilder>();
    auto maybe_builder = GetRootBuilderByTypeId(json_msg->type, *message_builder);
    if (!maybe_builder) {
      reportFailure("Unknown proto id: %lld", json_msg->type);
      return;
    }
    const std::string json_string = json_msg->data.dump();
    json_codec_->decode((::kj::StringPtr)json_string.c_str(), *maybe_builder);

    auto proto_message =
        std::make_shared<MallocProtoMessage>(std::move(message_builder), json_msg->type);
    proto_message->acqtime = json_msg->acqtime;
    proto_message->pubtime = json_msg->pubtime;
    proto_message->uuid = Uuid::Generate();
    // Message uuid is ignored intentionally as uuid is supposed to be unique.

    // Copy buffers
    for (const auto& buffer : json_msg->buffers) {
      proto_message->buffers.push_back(buffer.clone());
    }

    tx_proto().ledger()->provide({tx_proto().component(), tx_proto().tag()}, proto_message);
  }
}

}  // namespace alice
}  // namespace isaac
