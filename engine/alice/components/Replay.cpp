/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Replay.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/utils/utils.hpp"
#include "engine/core/byte.hpp"
#include "engine/core/logger.hpp"
#include "engine/gems/cask/cask.hpp"
#include "engine/gems/scheduler/scheduler.hpp"
#include "engine/gems/serialization/header.hpp"
#include "messages/alice.capnp.h"
#include "messages/uuid.hpp"

namespace isaac {
namespace alice {

Replay::Replay() {}
Replay::~Replay() {}

void Replay::loadLog() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  // Since not started replaying yet unset current time
  // Reset time_range_ for recalculation later in this function
  resetTimeRange();
  // clear channels data for previous log
  channels_.clear();
  std::string cask_directory = get_cask_directory();
  if (cask_directory == "") {
    LOG_WARNING("Empty replay log path to load!");
    return;
  }
  if (replayed_channels_.empty()) {
    LOG_WARNING("No channels added to replayed. Skipping loading '%s'", cask_directory.c_str());
    return;
  }
  cask_ = std::make_unique<cask::Cask>(cask_directory, cask::Cask::Mode::Read);
  readChannelIndex();
  for (const std::string& tag : replayed_channels_) {
    readChannelMessageHeaders(tag);
  }
}

void Replay::start() {
  startReplay();
}

void Replay::initialize() {
  message_ledger_ = node()->getComponent<MessageLedger>();

  // Connections to a replay node start message replay
  message_ledger_->addOnConnectAsTxCallback(
    [this](const MessageLedger::Endpoint& tx, const MessageLedger::Endpoint& rx) {
      if (tx.component != this) {
        return;
      }
      // start message replay
      this->addChannelReplay(tx.tag);
      // notify about replayed messages
      message_ledger_->addOnMessageCallback(tx, rx.component,
          [this, rx](ConstMessageBasePtr message) {
            message_ledger_->notifyScheduler(rx, message->pubtime);
          });
    });

  is_replaying_ = false;
  already_replayed_once_ = false;
}

void Replay::deinitialize() {
}

void Replay::startReplay() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!cask_) {
    loadLog();
  }
  if (!cask_) {
    return;
  }
  if (channels_.empty()) {
    LOG_WARNING("'%s': Zero channels found. Replay not started!", get_cask_directory().c_str());
    return;
  }
  if (replayed_channels_.empty()) {
    LOG_WARNING("No channels added to be replayed. Replay not started!");
    return;
  }
  // set scheduler reference time just before starting replay
  // Use the clock backend since the application will use it with the scheduler
  scheduler_reference_time_ = node()->clock()->timestamp();
  use_recorded_message_time_ = get_use_recorded_message_time();
  replay_time_offset_ = get_replay_time_offset();
  if (replay_time_offset_ < 0) {
    LOG_WARNING("Found replay_time_offset < 0. Resetting to 0.");
    replay_time_offset_ = 0;
  }
  if (already_replayed_once_ && use_recorded_message_time_) {
    LOG_WARNING(
        "Replaying more than once with recorded publish times from log for messages to be"
        " replayed may not work as expected. Try use_recorded_message_publish_time to false.");
  }
  already_replayed_once_ = true;
  is_replaying_ = true;
  for (const std::string& tag : replayed_channels_) {
    startChannelReplay(tag);
  }
}

void Replay::stopReplay() {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (!isReplaying()) return;  // replay already stopped
  is_replaying_ = false;
  // Remove all the scheduled tasks to prevent further replay.
  for (auto& kvp : channels_) {
    auto& channel_replay = kvp.second;
    channel_replay.replay_state = false;
    if (channel_replay.last_job_handle_) {
      node()->app()->backend()->scheduler()->destroyJob(*channel_replay.last_job_handle_);
      channel_replay.last_job_handle_ = std::nullopt;
    }
  }
}

void Replay::stop() {
  stopReplay();
}

void Replay::addChannelReplay(const std::string& tag) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  if (replayed_channels_.count(tag) != 0) {
    return;
  }
  replayed_channels_.insert(tag);
  if (cask_) {
    readChannelMessageHeaders(tag);
    // If channel added while a replay going on then start the channel's replay explicitly
    // else start function will take care of it.
    if (isReplaying()) startChannelReplay(tag);
  }
}

void Replay::readChannelIndex() {
  std::vector<uint8_t> blob;
  cask_->keyValueRead(Uuid::FromAsciiString("msg_chnl_idx"), blob);
  serialization::Header header;
  const uint8_t* segment_start = Deserialize(blob.data(), blob.data() + blob.size(), header);
  ASSERT(segment_start, "invalid header");
  ASSERT(header.segments.size() > 0, "invalid header");
  std::vector<kj::ArrayPtr<const ::capnp::word>> segments;
  for (uint16_t length : header.segments) {
    const uint8_t* segment_next = segment_start + length;
    segments.push_back(
        kj::ArrayPtr<const ::capnp::word>(reinterpret_cast<const ::capnp::word*>(segment_start),
                                          reinterpret_cast<const ::capnp::word*>(segment_next)));
    segment_start = segment_next;
  }
  ::capnp::SegmentArrayMessageReader reader(
      kj::ArrayPtr<const kj::ArrayPtr<const ::capnp::word>>(segments.data(), segments.size()));
  auto proto = reader.getRoot<MessageChannelIndexProto>();
  auto channels = proto.getChannels();
  for (size_t i = 0; i < channels.size(); i++) {
    const Uuid component_uuid = FromProto(channels[i].getComponentUuid());
    const std::string tag = channels[i].getTag();
    const Uuid series_uuid = FromProto(channels[i].getSeriesUuid());
    ASSERT(channels_.find(tag) == channels_.end(),
           "Found a duplicate channel tag '%s' in the log. This is currently not supported.");
    channels_[tag].series_uuid = series_uuid;
    // set replaying state as false for the channel
    channels_[tag].replay_state = false;
    LOG_DEBUG("Available channel in cask: '%s' (series: %s)", tag.c_str(), series_uuid.c_str());
  }
}

Replay::ChannelReplay& Replay::getChannelReplay(const std::string& tag) {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = channels_.find(tag);
  ASSERT(it != channels_.end(), "Channel '%s' not found in log", tag.c_str());
  return it->second;
}

const Replay::ChannelReplay& Replay::getChannelReplay(const std::string& tag) const {
  std::lock_guard<std::recursive_mutex> lock(mutex_);
  const auto it = channels_.find(tag);
  ASSERT(it != channels_.end(), "Channel '%s' not found in log", tag.c_str());
  return it->second;
}

void Replay::readChannelMessageHeaders(const std::string& tag) {
  ChannelReplay& channel_replay = getChannelReplay(tag);
  const Uuid series_uuid = channel_replay.series_uuid;
  cask_->seriesOpen(series_uuid);
  size_t count, value_size;
  cask_->seriesDimensions(series_uuid, count, value_size);
  if (count == 0) {
    return;
  }
  // Read the series
  const uint32_t flags = serialization::TIP_1_TIMESTAMP | serialization::TIP_2_UUID;
  auto& history = channel_replay.history;
  serialization::Header uuid_ts;
  std::vector<uint8_t> blob;
  for (size_t i = 0; i < count; i++) {
    cask_->seriesRead(series_uuid, i, blob);
    auto result = DeserializeWithoutTip(blob, flags, uuid_ts);
    ASSERT(result, "could not parse");
    history.push(*uuid_ts.timestamp, *uuid_ts.uuid);
  }
  // Update overall time range based on this channel's range, if required
  if (time_range_.start > history.oldest().stamp) {
    time_range_.start = history.oldest().stamp;
  }
  if (time_range_.end < history.youngest().stamp) {
    time_range_.end = history.youngest().stamp;
  }
  // Reset the offset to the start time if needed.
  if (get_replay_time_offset() < time_range_.start) {
    async_set_replay_time_offset(time_range_.start);
  }
  cask_->seriesClose(series_uuid);  // close since no more required for this channel

  LOG_DEBUG("Replay '%s': parsed %zu message headers for time range [%zd, %zd]", tag.c_str(),
            history.size(), history.oldest().stamp, history.youngest().stamp);
}

void Replay::startChannelReplay(const std::string& tag) {
  LOG_DEBUG("Started replay of channel '%s'", tag.c_str());
  getChannelReplay(tag).replay_state = true;  // indicate start of trying to replay
  // First message timestamp should be >= time offset
  enqueueNextMessage(tag, replay_time_offset_ - 1);
}

void Replay::enqueueNextMessage(const std::string& tag, int64_t prev_timestamp) {
  auto& channel_replay = getChannelReplay(tag);

  // Destroy the previous job
  if (channel_replay.last_job_handle_) {
    node()->app()->backend()->scheduler()->destroyJob(*channel_replay.last_job_handle_);
    channel_replay.last_job_handle_ = std::nullopt;
  }

  // If replay was explicitly stopped in between, no need to enqueue new messages
  if (!isReplaying()) return;

  // Find the next message to replay
  const auto maybe_message_entry = nextMessageToReplay(tag, prev_timestamp);
  if (!maybe_message_entry) {
    // Replay completed for the channel. Set replaying state to not replaying.
    channel_replay.replay_state = false;
    // If all channels completed replay and call stopReplay to reset/process required members
    std::unique_lock<std::mutex> lock(check_for_stop_mutex_);
    bool all_replay_completed = true;
    for (const auto& other_channel : channels_) {
      if (other_channel.second.replay_state) all_replay_completed = false;
    }
    if (all_replay_completed) {
      stopReplay();
      // Start replay again if looping is desired
      if (get_loop()) {
        startReplay();
      }
    }
    return;
  }
  const Uuid message_uuid = maybe_message_entry->first;
  const int64_t target_log_time = maybe_message_entry->second;

  // Create and schedule a new job for the next message
  scheduler::JobDescriptor job_descriptor;
  job_descriptor.execution_mode = scheduler::ExecutionMode::kOneShotTask;
  job_descriptor.target_start_time =
      target_log_time - replay_time_offset_ + scheduler_reference_time_;
  job_descriptor.slack = 0;
  job_descriptor.priority = 0;
  job_descriptor.event_trigger_limit = 1;
  job_descriptor.name = full_name() + "/" + tag;
  // Since replay spawns multiple disparate one shot jobs stat tracking
  // does not provide a meaningful signal and causes a lot of logging noise.
  job_descriptor.has_statistics = false;
  job_descriptor.action = [this, tag, message_uuid, target_log_time] {
    // Read the message from the log file and add it to the message ledger. This will trigger
    // message passing to other nodes.
    MessageBasePtr message = ReadMessageFromCask(message_uuid, *cask_);
    if (!use_recorded_message_time_) {
      message->pubtime = node()->clock()->timestamp();
      message->acqtime = message->acqtime - replay_time_offset_ + scheduler_reference_time_;
    }
    message_ledger_->provide({this, tag}, ConstMessageBasePtr(message));
    // Enqueue a job to replay the next message
    enqueueNextMessage(tag, target_log_time);
  };
  auto handle = node()->app()->backend()->scheduler()->createJobAndStart(job_descriptor);
  if (!handle) {
    LOG_ERROR("Could not schedule job. Replay of channel '%s' had ended prematurely", tag.c_str());
  }
  channel_replay.last_job_handle_ = handle;
}

void Replay::resetTimeRange() {
  time_range_.start = std::numeric_limits<int64_t>::max();
  time_range_.end = std::numeric_limits<int64_t>::lowest();
}

std::optional<std::pair<Uuid, int64_t>> Replay::nextMessageToReplay(
    const std::string& tag, int64_t previous_timestamp) const {
  const auto& history = getChannelReplay(tag).history;
  // Application time passed since replay started
  const int64_t app_time_since_start = node()->clock()->timestamp() - scheduler_reference_time_;
  // Log time corresponding to the current application time
  const int64_t log_time = app_time_since_start - replay_time_offset_;
  // Find the corresponding index
  size_t index = static_cast<size_t>(history.upper_index(log_time));
  // Guarantee that we have a new message
  while (index < history.size() && history.at(index).stamp <= previous_timestamp) index++;
  // Return the UUID and timestamp of the next message we should replay
  if (index >= history.size()) {
    return std::nullopt;
  }
  const auto message_entry = history.at(index);
  return std::pair<Uuid, int64_t>{message_entry.state, message_entry.stamp};
}

}  // namespace alice
}  // namespace isaac
