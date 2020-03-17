/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "RecorderBridge.hpp"

#include <errno.h>
#include <unistd.h>

#include <iomanip>
#include <sstream>
#include <string>
#include <utility>

#include "engine/gems/system/filesystem.hpp"

namespace isaac {
namespace alice {

namespace {
constexpr char kRequestRecCmd[] = "recording";
constexpr char kStartParam[] = "true";
constexpr char kStopParam[] = "false";
constexpr char kRecordingReply[] = "recording";
constexpr char kStopReply[] = "stopped";
// Error related string
constexpr char kNoError[] = "No error";
constexpr char kNoChannelError[] = "There are no channels connected to the Recorder component";
constexpr char kInvalidPathError[] = "Invalid path for log";
}  // namespace

void RecorderBridge::start() {
  app_uuid_ = node()->app()->uuid().str();
  tickPeriodically();
  records_number_ = 0;
  recording_time_ = 0;
  tag_suffix_ = "";
}

void RecorderBridge::stop() {
  recorder_component_ = nullptr;
}

void RecorderBridge::tick() {
  // Late initialization, we need to do it here because Record node might not be ready
  // when we execute our start method
  if (!recorder_component_) {
    recorder_component_ =
        node()->app()->findComponentByName<Recorder>(get_recorder_component_name());
    ASSERT(recorder_component_, "expected isaac::alice::Recorder* type pointer");
    app_start_time_ = node()->clock()->timestamp();
    if (recorder_component_->get_enabled()) {
      records_number_ = 1;
      start_time_ = node()->clock()->timestamp() - app_start_time_;
    } else {
      start_time_ = -1;  // Sentinal value: not started
    }
    base_path_ = recorder_component_->get_base_directory();
    tag_ = recorder_component_->get_tag();
    if (recorder_component_->numChannels() == 0) {
      LOG_WARNING("Recording disabled since no channels are connected to Recorder component");
    }
  }

  // Process all messages from sight since last tick function call
  // TODO It is unclear why `this->` is required explicitly by the compiler in the lambda function
  rx_request().processAllNewMessages([this](auto json, int64_t pubtime, int64_t acqtime) {
    if (this->recorder_component_->numChannels() == 0) {
      this->sendNotice(kStopReply, kNoChannelError);
    } else if (this->recorder_component_->get_enabled()) {
      if (this->isMessageValid(json, kRequestRecCmd, kStopParam)) {
        // Stop recording logic
        this->recorder_component_->async_set_enabled(false);
        this->sendNotice(kStopReply, kNoError);
      }
    } else {
      if (this->isMessageValid(json, kRequestRecCmd, kStartParam)) {
        // Received a start recording signal
        // Start a new recording logic
        // First, we get parameters from front end
        base_path_ = serialization::TryGetFromMap<std::string>(json, "base_path")->c_str();
        tag_ = serialization::TryGetFromMap<std::string>(json, "tag")->c_str();
        // Now we signal component to start new recording
        this->recorder_component_->async_set_base_directory(this->base_path_);
        this->recorder_component_->async_set_tag(this->tag_ + this->getSuffix());
        if (this->validatePath()) {
          this->recorder_component_->openCask();
          this->recorder_component_->async_set_enabled(true);
          // Finally, update our state
          this->records_number_++;
          this->start_time_ = this->node()->clock()->timestamp() - this->app_start_time_;
          this->recording_time_ = 0;
          this->sendNotice(kRecordingReply, kNoError);
        } else {
          const std::string root = base_path_ + "/" + this->tag_ + this->getSuffix();
          LOG_ERROR("Invalid path for log: %s", root.c_str());
          this->sendNotice(kStopReply, kInvalidPathError);
        }
      }
    }
  });
  // Broadcast current state again or in case no messages were received
  if (recorder_component_->get_enabled()) {
    recording_time_ = node()->clock()->timestamp() - (app_start_time_ + start_time_);
    sendNotice(kRecordingReply, kNoError);
  } else {
    sendNotice(kStopReply, kNoError);
  }
}

bool RecorderBridge::validatePath() {
  const std::string root = base_path_ + "/" + app_uuid_ + "/" + tag_ + getSuffix();
  // Validate that you can write in base path and that root does not exist
  return IsValidWriteDirectory(base_path_) && !IsValidDirectory(root);
}

std::string RecorderBridge::getSuffix() {
  std::stringstream number_formater;
  number_formater << std::setw(get_count_width()) << std::setfill('0') << records_number_;
  return number_formater.str();
}

bool RecorderBridge::isMessageValid(const Json& json, const std::string& expected_cmd,
    const std::string& expected_param) {
  const auto maybe_cmd = serialization::TryGetFromMap<std::string>(json, "cmd");
  if (!maybe_cmd) {
      LOG_ERROR("Received message does not contain 'cmd': %s", json.dump(2).c_str());
      return false;
  }
  if (*maybe_cmd != expected_cmd) {
      LOG_ERROR("Unexpected cmd: %s", maybe_cmd->c_str());
      return false;
  }
  const auto param = serialization::TryGetFromMap<std::string>(json, "cmdparams");
  if (!param) {
      LOG_ERROR("Received cmd: %s does not contain cmdparams", maybe_cmd->c_str());
      return false;
  }
  if (*param != expected_param) {
      LOG_ERROR("Unexpected param %s for cmd: %s", param->c_str(), maybe_cmd->c_str());
      return false;
  }

  return true;
}

void RecorderBridge::sendNotice(const std::string& cmd, const std::string& error_msg) {
  Json msg;
  msg["cmd"] = cmd;
  // Since we work in nanoseconds and JavaScript requires integer miliseconds
  msg["start_time"] = static_cast<int>(ToSeconds(start_time_) * 1000.0);
  msg["current_time"] = static_cast<int>(ToSeconds(recording_time_) * 1000.0);
  msg["base_directory"] = base_path_;
  msg["tag"] = tag_;
  std::string tag = recorder_component_->get_tag();
  std::string base_path = recorder_component_->get_base_directory();
  // This needs to mimic the way Recorder builds its cask
  msg["absolute_path"] = base_path + "/" + app_uuid_ + (tag.empty() ? "" : "/" + tag);
  msg["app_uuid"] = app_uuid_;
  // Append the error state and message
  msg["error_state"] = error_msg != kNoError;
  msg["error_msg"] = error_msg;
  tx_reply().publish(std::move(msg));
}

}  // namespace alice
}  // namespace isaac
