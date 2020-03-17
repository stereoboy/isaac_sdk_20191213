/*
Copyright (c) 2018, 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "PoseTree.hpp"

namespace isaac {
namespace alice {

void PoseTree::start() {
  reportSuccess();  // do not participate in status updates
}

Pose3d PoseTree::get(const FrameId& lhs, const FrameId& rhs, double time) const {
  auto maybe = tryGet(lhs, rhs, time);
  ASSERT(maybe, "Could not get the transformation %s_T_%s.", lhs.c_str(), rhs.c_str());
  return *maybe;
}

Pose2d PoseTree::getPose2XY(const FrameId& lhs, const FrameId& rhs, double time) const {
  return get(lhs, rhs, time).toPose2XY();
}

std::optional<Pose3d> PoseTree::tryGet(const FrameId& lhs, const FrameId& rhs, double time) const {
  const Uuid lhs_uuid = Uuid::FromAsciiString(lhs);
  const Uuid rhs_uuid = Uuid::FromAsciiString(rhs);
  return tryGet(lhs_uuid, rhs_uuid, time);
}

std::optional<Pose2d> PoseTree::tryGetPose2XY(const FrameId& lhs, const FrameId& rhs,
                                              double time) const {
  auto maybe = tryGet(lhs, rhs, time);
  if (maybe) {
    return maybe->toPose2XY();
  } else {
    return std::nullopt;
  }
}

std::optional<Pose3d> PoseTree::tryGet(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  std::shared_lock<std::shared_timed_mutex> lock(pose_tree_mutex_);
  return pose_tree_.get(lhs, rhs, stamp);
}

bool PoseTree::set(const FrameId& lhs, const FrameId& rhs, const Pose3d& lhs_T_rhs, double time) {
  const Uuid lhs_uuid = Uuid::FromAsciiString(lhs);
  const Uuid rhs_uuid = Uuid::FromAsciiString(rhs);
  return set(lhs_uuid, rhs_uuid, time, lhs_T_rhs);
}

bool PoseTree::set(const FrameId& lhs, const FrameId& rhs, const Pose2d& lhs_T_rhs, double time) {
  return set(lhs, rhs, Pose3d::FromPose2XY(lhs_T_rhs), time);
}

bool PoseTree::set(const Uuid& lhs, const Uuid& rhs, double stamp, const Pose3d& lhs_T_rhs) {
  std::unique_lock<std::shared_timed_mutex> pose_tree_lock(pose_tree_mutex_);
  const bool result = pose_tree_.set(lhs, rhs, stamp, lhs_T_rhs);
  pose_tree_lock.unlock();

  std::shared_lock<std::shared_timed_mutex> callbacks_lock(callbacks_mutex_);
  for (auto& kvp : callbacks_) {
    kvp.second(lhs, rhs, stamp, lhs_T_rhs);
  }
  callbacks_lock.unlock();

  return result;
}

pose_tree::PoseTree PoseTree::clonePoseTree() const {
  std::shared_lock<std::shared_timed_mutex> lock(pose_tree_mutex_);
  return pose_tree_;
}

pose_tree::PoseTree PoseTree::cloneLatestPoseTree() const {
  std::shared_lock<std::shared_timed_mutex> lock(pose_tree_mutex_);
  return pose_tree_.latest();
}

void PoseTree::registerForUpdates(const Component* source, UpdateFunction callback) {
  std::unique_lock<std::shared_timed_mutex> lock(callbacks_mutex_);
  callbacks_[source] = callback;
}

void PoseTree::deregisterForUpdates(const Component* source) {
  std::unique_lock<std::shared_timed_mutex> lock(callbacks_mutex_);
  callbacks_.erase(source);
}

}  // namespace alice
}  // namespace isaac
