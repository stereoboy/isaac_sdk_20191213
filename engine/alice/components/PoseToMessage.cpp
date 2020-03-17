/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/components/PoseToMessage.hpp"

#include <string>

#include "messages/math.hpp"

namespace isaac {
namespace alice {

void PoseToMessage::start() {
  tickPeriodically();
}

void PoseToMessage::tick() {
  // Read pose from pose tree
  const std::string lhs_frame = get_lhs_frame();
  const std::string rhs_frame = get_rhs_frame();
  const auto maybe_pose = node()->pose().tryGet(lhs_frame, rhs_frame, getTickTime());
  if (!maybe_pose) {
    return;
  }

  // Publish message
  auto pose_proto = tx_pose().initProto();
  pose_proto.setLhs(lhs_frame);
  pose_proto.setRhs(rhs_frame);
  ToProto(*maybe_pose, pose_proto.initPose());
  tx_pose().publish();
}

}  // namespace alice
}  // namespace isaac
