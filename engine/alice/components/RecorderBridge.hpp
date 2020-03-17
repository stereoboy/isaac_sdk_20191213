/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>

#include "engine/alice/application.hpp"
#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/components/Recorder.hpp"

namespace isaac {
namespace alice {

// @internal
// Communication Bridge between WebsightServer and Record Node
class RecorderBridge : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;

  // NUmber of trailing zeros in the output folder name
  ISAAC_PARAM(int, count_width, 3);
  // Component name in format node/component. Example: replay/isaac.alice.Recorder
  ISAAC_PARAM(std::string, recorder_component_name);

  // Request to replay node
  ISAAC_RAW_RX(nlohmann::json, request);
  // Reply from replay node
  ISAAC_RAW_TX(nlohmann::json, reply);

 private:
  // Send reply to the Front End
  void sendNotice(const std::string& cmd, const std::string& error_msg);
  // Validate an incomming msg from the Front End
  bool isMessageValid(const Json& json, const std::string& expected_cmd,
      const std::string& expected_param);
  // Create a custom suffix for a tag
  std::string getSuffix();
  // Validate the desired log path
  bool validatePath();

  // Log base directory
  std::string base_path_;
  // Tag identifier
  std::string tag_;
  // Since internally, we have our own tag convention
  std::string tag_suffix_;
  // Ptr to the recorder component
  Recorder* recorder_component_;
  // The application Uuid
  std::string app_uuid_;
  // The start time of the current/last recording
  int64_t start_time_;
  // The start time of the app execution
  int64_t app_start_time_;
  // The number of recordings during the execution
  size_t records_number_;
  // The current time of the recording
  int64_t recording_time_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::RecorderBridge);
