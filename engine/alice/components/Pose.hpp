/*
Copyright (c) 2018, 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>

#include "engine/alice/component.hpp"
#include "engine/core/math/pose2.hpp"
#include "engine/core/math/pose3.hpp"
#include "engine/core/optional.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace alice {

class PoseTree;

// Provides convenience functions to access 3D transformations from the application wide pose tree.
//
// This component is added to every node by default and does not have to be added manually.
//
// Poses use 64-bit floating point types and are 3-dimensional. All coordinate frames for the whole
// application are stored in a single central pose tree.
//
// All functions below accept two coordinate frames: `lhs` and `rhs`. This refers to the pose
// lhs_T_rhs which is the relative transformations between these two coordinate frames. In
// particular the following equations hold:
//   p_lhs = lhs_T_rhs * p_rhs
//   a_T_c = a_T_b * b_T_c
//
// Not all coordinate frames are connected. If this is the case or either of the two coordinate
// frames does not exist the pose is said to be "invalid".
class Pose : public Component {
 public:
  void start() override;

  using FrameId = std::string;

  // Gives the relative pose `lhs_T_rhs` between frame `lhs` and `rhs`. This function will assert
  // if the pose is invalid.
  Pose3d get(const FrameId& lhs, const FrameId& rhs, double time) const;
  // Similar to `get`, but also converts the 3D pose to a 2D pose relative to the plane Z = 0.
  Pose2d getPose2XY(const FrameId& lhs, const FrameId& rhs, double time) const;

  // Gives the relative pose `lhs_T_rhs` between frame `lhs` and `rhs`. This function will return
  // nullopt if the pose is invalid.
  std::optional<Pose3d> tryGet(const FrameId& lhs, const FrameId& rhs, double time) const;
  // Similar to `tryGet`, but also converts the 3D pose to a 2D pose relative to the plane Z = 0.
  std::optional<Pose2d> tryGetPose2XY(const FrameId& lhs, const FrameId& rhs, double time) const;
  // Gets the latest 3D pose lhs_T_rhs for the given time
  std::optional<Pose3d> tryGet(const Uuid& lhs, const Uuid& rhs, double stamp) const;

  // Sets the relative pose between two coordinate frames.  If the parameter is not specified and
  // the pose could not be set the function will return false.
  bool set(const FrameId& lhs, const FrameId& rhs, const Pose3d& lhs_T_rhs, double time);
  // Similar to `set` but for setting a pose in the Z = 0 plane.
  bool set(const FrameId& lhs, const FrameId& rhs, const Pose2d& lhs_T_rhs, double time);
  // Sets 3D pose lhs_T_rhs (and rhs_T_lhs) for the given time
  bool set(const Uuid& lhs, const Uuid& rhs, double stamp, const Pose3d& lhs_T_rhs);

 private:
  PoseTree* pose_tree_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::Pose)
