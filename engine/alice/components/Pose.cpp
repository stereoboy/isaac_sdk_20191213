/*
Copyright (c) 2018, 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Pose.hpp"

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/components/PoseTree.hpp"

namespace isaac {
namespace alice {

void Pose::start() {
  pose_tree_ = node()->app()->backend()->pose_tree();
  reportSuccess();  // do not participate in status updates
}

Pose3d Pose::get(const FrameId& lhs, const FrameId& rhs, double time) const {
  auto maybe = tryGet(lhs, rhs, time);
  ASSERT(maybe, "Could not get the transformation %s_T_%s.", lhs.c_str(), rhs.c_str());
  return *maybe;
}

Pose2d Pose::getPose2XY(const FrameId& lhs, const FrameId& rhs, double time) const {
  return get(lhs, rhs, time).toPose2XY();
}

std::optional<Pose3d> Pose::tryGet(const FrameId& lhs, const FrameId& rhs, double time) const {
  const Uuid lhs_uuid = Uuid::FromAsciiString(lhs);
  const Uuid rhs_uuid = Uuid::FromAsciiString(rhs);
  return tryGet(lhs_uuid, rhs_uuid, time);
}

std::optional<Pose2d> Pose::tryGetPose2XY(const FrameId& lhs, const FrameId& rhs,
                                          double time) const {
  auto maybe = tryGet(lhs, rhs, time);
  if (maybe) {
    return maybe->toPose2XY();
  } else {
    return std::nullopt;
  }
}

std::optional<Pose3d> Pose::tryGet(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  return pose_tree_->tryGet(lhs, rhs, stamp);
}

bool Pose::set(const FrameId& lhs, const FrameId& rhs, const Pose3d& lhs_T_rhs, double time) {
  const Uuid lhs_uuid = Uuid::FromAsciiString(lhs);
  const Uuid rhs_uuid = Uuid::FromAsciiString(rhs);
  return set(lhs_uuid, rhs_uuid, time, lhs_T_rhs);
}

bool Pose::set(const FrameId& lhs, const FrameId& rhs, const Pose2d& lhs_T_rhs, double time) {
  return set(lhs, rhs, Pose3d::FromPose2XY(lhs_T_rhs), time);
}

bool Pose::set(const Uuid& lhs, const Uuid& rhs, double stamp, const Pose3d& lhs_T_rhs) {
  return pose_tree_->set(lhs, rhs, stamp, lhs_T_rhs);
}

}  // namespace alice
}  // namespace isaac
