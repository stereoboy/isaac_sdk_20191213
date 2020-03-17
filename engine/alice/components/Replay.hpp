/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <utility>

#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/message.hpp"
#include "engine/gems/algorithm/timeseries.hpp"
#include "engine/gems/scheduler/job_descriptor.hpp"

namespace isaac { namespace cask { class Cask; }}

namespace isaac {
namespace alice {

// Replays data from a log file which was recorded by a Recorder component. See the documentation
// for the Recorder component for more information.
class Replay : public Component {
 public:
  // Struct to store start and end times of a single time range
  struct TimeRange {
    // The start timestamp of the range
    int64_t start;
    // The end timestamp of the range
    int64_t end;
  };
  Replay();
  ~Replay();

  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // Starts replay of a loaded log
  void startReplay();
  // Stops an ongoing replay
  void stopReplay();
  // Opens cask from root and parses contents without starting replay
  void loadLog();
  // Getter for log loaded state
  bool isLogLoaded() const { return !(cask_ == nullptr); }
  // Getter for time range
  const TimeRange& getTimeRange() const { return time_range_; }
  // Getter for current time being replayed
  int64_t getReplayTime() const {
    return isReplaying()
               ? node()->clock()->timestamp() - scheduler_reference_time_ + replay_time_offset_
               : -1;
  }
  // Getter for log replaying state
  bool isReplaying() const { return is_replaying_; }

  // The cask directory used to replay data from
  ISAAC_PARAM(std::string, cask_directory, "");
  // Time offset to start a replay from between a log
  ISAAC_PARAM(int64_t, replay_time_offset, 0);
  // Decides whether to use recorded message pubtime and acqtime or replay current time as pubtime
  // and synchronize the acqtime using the starting time of the replay.
  ISAAC_PARAM(bool, use_recorded_message_time, false);
  // If this is enabled replay will start from the beginning when it was replayed
  ISAAC_PARAM(bool, loop, false);

 private:
  // Struct to store information of a channel to be replayed
  struct ChannelReplay {
    // UUID for the series used in the log file
    Uuid series_uuid;
    // Message history of all messages from the channel
    Timeseries<Uuid, int64_t> history;
    // Indicates wether this channel is currently replaying
    bool replay_state;
    // scheduler job handle for the previous job
    std::optional<scheduler::JobHandle> last_job_handle_;
  };

  // Adds a channel for replaying
  void addChannelReplay(const std::string& tag);
  // Reads the channel index from the log file
  void readChannelIndex();
  // Gets the channel replay object for a channel
  ChannelReplay& getChannelReplay(const std::string& tag);
  const ChannelReplay& getChannelReplay(const std::string& tag) const;
  // Starts replaying a channel
  void startChannelReplay(const std::string& tag);
  // Gets the next message for a channel and enqeueus it for replay
  void enqueueNextMessage(const std::string& tag, int64_t prev_timestamp);
  // Parses message headers from the channel with tag and updates global time ranges
  void readChannelMessageHeaders(const std::string& tag);
  // Resets time range container values
  void resetTimeRange();
  // Finds the next message to replay on a channel; returns std::nullopt in case there are no
  // more messages to replay.
  std::optional<std::pair<Uuid, int64_t>> nextMessageToReplay(const std::string& tag,
                                                              int64_t previous_timestamp) const;

  // General mutex for thread safety
  mutable std::recursive_mutex mutex_;
  // Mutex to check if we need to stop. Necessary to avoid race conditions when stopping or
  // looping the replay.
  std::mutex check_for_stop_mutex_;

  bool is_replaying_;  // state of replaying activity
  int64_t scheduler_reference_time_;  // stores scheduler time before start of a replay
  std::unique_ptr<cask::Cask> cask_;
  bool already_replayed_once_;
  TimeRange time_range_;  // global time range inclusive of all channels
  std::map<std::string, ChannelReplay> channels_;  // channels found in the loaded log
  std::set<std::string> replayed_channels_;  // channels connected to the node and to be replayed

  MessageLedger* message_ledger_;  // cached for performance

  int64_t replay_time_offset_;  // saves the parameter to keep it constant after start
  bool use_recorded_message_time_;  // saves the parameter to keep it constant after start
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::Replay)
