/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/alice_codelet.hpp"
#include "engine/gems/state/io.hpp"

namespace isaac {
namespace kinova_jaco {

// A sample controller for the Jaco arm that sends cartesian pose commands to move end effector
// to the home pose.
class KinovaJacoSampleController : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override {}

  // Command for end effector position and orientation
  ISAAC_PROTO_TX(StateProto, cartesian_pose_command);

  // 3D pose of the home configuration for the Jaco arm
  ISAAC_PARAM(Pose3d, home_pose, Pose3d({SO3d::FromQuaternion({0.5511,      // qw
                                                               0.6452,      // qx
                                                               0.3175,      // qy
                                                               0.4234 }),   // qz
                                                              {0.2116,      // px
                                                              -0.2648,      // py
                                                               0.5054}}));  // pz
};

}  // namespace kinova_jaco
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::kinova_jaco::KinovaJacoSampleController);
