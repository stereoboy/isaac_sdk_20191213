/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "AliceSight.hpp"

#include <string>
#include <utility>

#include "engine/alice/alice.hpp"
#include "engine/gems/sight/sight.hpp"

namespace isaac {
namespace sight {

void AliceSight::start() {
  // Setup raw sight so that it uses alice's websight
  ResetSight(this);
}

void AliceSight::stop() {
  // Unregisters raw sight
  ResetSight(nullptr);
}

void AliceSight::plotValue(const std::string& name, int64_t timestamp, float value) {
  show(name, timestamp, value);
}

void AliceSight::plotValue(const std::string& name, int64_t timestamp, double value) {
  show(name, timestamp, value);
}

void AliceSight::plotValue(const std::string& name, int64_t timestamp, int value) {
  show(name, timestamp, value);
}

void AliceSight::plotValue(const std::string& name, int64_t timestamp, int64_t value) {
  show(name, timestamp, value);
}

void AliceSight::log(const char* file, int line, logger::Severity severity, const char* log,
                     int64_t timestamp) {
  // FIXME implement
}

void AliceSight::drawCanvas(const std::string& name, sight::Sop sop) {
  show(name, std::move(sop));
}

}  // namespace sight
}  // namespace isaac
