/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "ReplayBridge.hpp"

#include <fstream>
#include <string>

#include "engine/alice/application.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/components/bridge_utils.hpp"
#include "engine/gems/system/filesystem.hpp"

namespace isaac {
namespace alice {

namespace {
  // Command Strings
  const std::string kLoadCmd = "load";
  const std::string kStartCmd = "start";
  const std::string kStopCmd = "stop";
  const std::string kRangeUpdateCmd = "range-update";
  const std::string kTimeUpdateCmd = "time-update";
  // Field Strings
  const std::string kLogPathField = "log-path";
  const std::string kStartTimeField = "start-time";
  const std::string kEndTimeField = "end-time";
  const std::string kTimeField = "time";
}  // namespace

void ReplayBridge::start() {
  tickPeriodically();
}

void ReplayBridge::stop() {
  replay_component_ = nullptr;
}


void ReplayBridge::tick() {
  // Set replay component here since by the end of our start replay node might have not started
  // based on Node start execution sequence of the application
  if (!replay_component_) {
    auto* component = node()->app()->findComponentByName(get_replay_component_name());
    if (!component) {
      LOG_WARNING("Component '%s' not found. Returning early", get_replay_component_name().c_str());
      return;
    }
    replay_component_ = dynamic_cast<Replay*>(component);
    ASSERT(replay_component_, "expected isaac::alice::Replay* type pointer");
  }
  // Process all messages from sight since last tick function call
  // TODO It is unclear why `this->` is required explicitly by the compiler in the lambda function
  rx_request().processAllNewMessages([this](auto json, int64_t pubtime, int64_t acqtime) {
    const auto maybe_cmd = serialization::TryGetFromMap<std::string>(json, "cmd");
    if (!maybe_cmd) {
      LOG_ERROR("Recieved message does not contain 'cmd': %s", json.dump(2).c_str());
      return;
    }
    if (*maybe_cmd == kLoadCmd) {
      const auto maybe_cmd_params = serialization::TryGetFromMap<std::string>(json, "cmd-params");
      if (!maybe_cmd_params) {
        LOG_ERROR("Expected 'cmd-params'(string) with 'cmd': %s", json.dump(2).c_str());
        return;
      } else {
        if (IsValidPath(*maybe_cmd_params)) {
          this->replay_component_->async_set_cask_directory(*maybe_cmd_params);
          this->replay_component_->loadLog();
        } else {
          LOG_ERROR("Invalid log directory: %s", (*maybe_cmd_params).c_str());
        }
      }
    } else if (*maybe_cmd == kStartCmd) {
      const auto maybe_cmd_params = serialization::TryGetFromMap<int64_t>(json, "cmd-params");
      if (!maybe_cmd_params) {
        LOG_ERROR("Expected 'cmd-params'(int64_t) with 'cmd': %s", json.dump(2).c_str());
        return;
      } else {
        this->replay_component_->async_set_replay_time_offset(*maybe_cmd_params);
        this->replay_component_->startReplay();
      }
    } else if (*maybe_cmd == kStopCmd) {
      this->replay_component_->stopReplay();
    } else {
      LOG_ERROR("Unknown cmd: %s", maybe_cmd->c_str());
    }
  });
  // Based on replay component's state send necessary
  // info to front-end periodically
  Json load_cmd_reply_params;
  load_cmd_reply_params[kLogPathField] = "";
  if (replay_component_->isLogLoaded()) {
    load_cmd_reply_params[kLogPathField] = replay_component_->get_cask_directory();
  }
  SendMsgToWebsightServer(tx_reply(), kLoadCmd, load_cmd_reply_params);
  // Replay computes log's time range upon load. So start sending the range. Since new channels
  // maybe added dynamically, thus, changing the overall time range, hence, continuously send range.
  const Replay::TimeRange& time_range = replay_component_->getTimeRange();
  Json range_update_cmd_reply_params;
  range_update_cmd_reply_params[kStartTimeField] = time_range.start;
  range_update_cmd_reply_params[kEndTimeField] = time_range.end;
  SendMsgToWebsightServer(tx_reply(), kRangeUpdateCmd, range_update_cmd_reply_params);
  // Send current replay time
  Json time_update_cmd_reply_params;
  time_update_cmd_reply_params[kTimeField] = replay_component_->getReplayTime();
  SendMsgToWebsightServer(tx_reply(), kTimeUpdateCmd, time_update_cmd_reply_params);
}

bool ReplayBridge::IsValidPath(const std::string& path) {
  const std::string extra_path = path + "/kv";
  // validate that path is readable and that contains a kv subdirecory
  return IsValidReadDirectory(path) && IsValidDirectory(extra_path);
}

}  // namespace alice
}  // namespace isaac
