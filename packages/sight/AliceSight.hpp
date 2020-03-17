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

#include "engine/alice/alice_codelet.hpp"
#include "engine/gems/sight/sight_interface.hpp"

namespace isaac {
namespace sight {

// Interface for sight.
// Provide a default implementation which does nothing.
class AliceSight : public alice::Codelet, public SightInterface {
 public:
  void start() override;
  void stop() override;

  void plotValue(const std::string& name, int64_t timestamp, float value) override;
  void plotValue(const std::string& name, int64_t timestamp, double value) override;
  void plotValue(const std::string& name, int64_t timestamp, int value) override;
  void plotValue(const std::string& name, int64_t timestamp, int64_t value) override;
  void log(const char* file, int line, logger::Severity severity, const char* log,
           int64_t timestamp) override;
  void drawCanvas(const std::string& name, sight::Sop sop) override;
};

}  // namespace sight
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::sight::AliceSight);
