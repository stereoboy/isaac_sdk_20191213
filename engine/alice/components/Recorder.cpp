/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Recorder.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/utils/utils.hpp"
#include "engine/core/assert.hpp"
#include "engine/gems/cask/cask.hpp"
#include "engine/gems/serialization/blob.hpp"
#include "engine/gems/serialization/header.hpp"
#include "messages/alice.capnp.h"
#include "messages/uuid.hpp"

namespace isaac {
namespace alice {

Recorder::Recorder() {}
Recorder::~Recorder() {}

void Recorder::initialize() {
  // Messages received by Recorder will be written to the log
  MessageLedger* ledger = node()->getComponent<MessageLedger>();
  ASSERT(ledger, "Replay requires MessageLedger component");
  ledger->addOnConnectAsRxCallback(
    [ledger, this](const MessageLedger::Endpoint& tx, const MessageLedger::Endpoint& rx) {
      if (rx.component != this) {
        return;
      }
      // notify about replayed messages
      ledger->addOnMessageCallback(rx, tx.component,
          [this, tx, rx](ConstMessageBasePtr message) {
            this->log(tx.component, rx.tag, message);
          });
    });
}

void Recorder::start() {
  // TODO putting this here will not allow us to record data before starting the network
  openCask();
}

void Recorder::stop() {
  // TODO maybe close cask once no recorder needs it anymore
}

void Recorder::deinitialize() {
}

void Recorder::openCask() {
  std::string root = get_base_directory() + "/" + node()->app()->uuid().str();
  const std::string log_tag = get_tag();
  if (!log_tag.empty()) {
    root = root + "/" + log_tag;
  }
  // Close old cask before starting a new one
  if (cask_) {
    ASSERT(cask_->getRoot() != root, "Recorder cannot reuse the same log directory");
    for (auto& it : component_key_to_uuid_) {
      cask_->seriesClose(it.second);
    }
    component_key_to_uuid_.clear();
    cask_->close();
  }
  cask_ = std::make_unique<cask::Cask>(root, cask::Cask::Mode::Write);
}

void Recorder::log(const Component* component, const std::string& key,
                   ConstMessageBasePtr message) {
  logMessage(component, message);
  logChannel(component, key, message);
}

void Recorder::logMessage(const Component* component, ConstMessageBasePtr message) {
  if (!get_enabled()) {
    return;
  }
  if (!cask_) {
    LOG_ERROR("Logging before start is currently not supported");
    return;
  }
  WriteMessageToCask(message, *cask_);
}

void Recorder::logChannel(const Component* component, const std::string& key,
                          ConstMessageBasePtr message) {
  if (!get_enabled()) {
    return;
  }
  if (!cask_) {
    LOG_ERROR("Logging before start is currently not supported");
    return;
  }
  // add to series
  auto compkey = std::pair<Uuid, std::string>{component->uuid(), key};
  auto it = component_key_to_uuid_.find(compkey);
  if (it == component_key_to_uuid_.end()) {
    Uuid uuid = Uuid::Generate();
    it = component_key_to_uuid_.insert({compkey, uuid}).first;
    cask_->seriesOpen(uuid, 24);
    writeChannelIndex();
  }
  serialization::Header uuid_ts;
  uuid_ts.timestamp = message->pubtime;
  uuid_ts.uuid = message->uuid;
  std::array<uint8_t, 24> keybytes;
  const uint32_t flags = serialization::TIP_1_TIMESTAMP | serialization::TIP_2_UUID;
  SerializeWithoutTip(uuid_ts, flags, keybytes.data(), keybytes.data() + keybytes.size());
  cask_->seriesAppend(it->second, ByteArrayConstView{keybytes.data(), keybytes.size()});
}

void Recorder::writeChannelIndex() {
  // the proto
  ::capnp::MallocMessageBuilder header_builder;
  auto index = header_builder.initRoot<MessageChannelIndexProto>();
  auto channels = index.initChannels(component_key_to_uuid_.size());
  size_t counter = 0;
  for (const auto& kvp : component_key_to_uuid_) {
    auto channel = channels[counter++];
    ToProto(kvp.first.first, channel.initComponentUuid());
    channel.setTag(kvp.first.second);
    ToProto(kvp.second, channel.initSeriesUuid());
  }
  std::vector<ByteArrayConstView> blobs;
  serialization::CapnpArraysToBlobs(header_builder.getSegmentsForOutput(), blobs);
  // the header
  serialization::Header header;
  header.segments.reserve(blobs.size());
  for (const auto& blob : blobs) {
    header.segments.push_back(blob.size());
  }
  std::vector<uint8_t> buffer;
  Serialize(header, buffer);
  // write all segments (including the header)
  blobs.insert(blobs.begin(), ByteArrayConstView{buffer.data(), buffer.size()});
  cask_->keyValueWrite(Uuid::FromAsciiString("msg_chnl_idx"), blobs);
}

size_t Recorder::numChannels() {
  MessageLedger* ledger = node()->getComponent<MessageLedger>();
  return ledger ? ledger->numSourceChannels() : 0;
}

}  // namespace alice
}  // namespace isaac
