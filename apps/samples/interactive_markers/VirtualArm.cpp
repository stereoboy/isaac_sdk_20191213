/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "VirtualArm.hpp"

#include <string>
#include <vector>

#include "engine/alice/components/Pose.hpp"

namespace {
const char kLeftShoulder[] = "leftShoulder";
const char kLeftElbow[] = "leftElbow";
}  // namespace

void VirtualArm::start() {
  tickPeriodically();
  counter_ = 0;
}

void VirtualArm::tick() {
  const isaac::Pose3d shoulder_T_elbow{
      isaac::SO3d::FromAngleAxis(isaac::Pi<double> * (counter_ % 11) / 5.0, {0.0, 0.0, 1.0}),
      isaac::Vector3d(0.0, 0.0, 1.0)};
  node()->pose().set(kLeftShoulder, kLeftElbow, shoulder_T_elbow, getTickTime());
  counter_++;
}
