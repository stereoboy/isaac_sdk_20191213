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

#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/components/Replay.hpp"

namespace isaac {
namespace alice {

// Communication Bridge between WebsightServer and Replay Node
class ReplayBridge : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;

  // Replay component name in format node/component. Ex: replay/isaac.alice.Replay
  ISAAC_PARAM(std::string, replay_component_name);

  // Request to replay node
  ISAAC_RAW_RX(nlohmann::json, request);
  // Reply from replay node
  ISAAC_RAW_TX(nlohmann::json, reply);

 private:
  // Validate the desired log path
  static bool IsValidPath(const std::string& path);

  Replay* replay_component_;  // ptr to the controlled replay component
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ReplayBridge);
