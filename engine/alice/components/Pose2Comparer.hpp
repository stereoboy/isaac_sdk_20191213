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
#include "messages/messages.hpp"

namespace isaac {
namespace alice {

// Reports success when the two poses are within threshold
class Pose2Comparer : public alice::Codelet {
 public:
  void start() override;
  void tick() override;

  // Name of the left hand side frame of the first pose
  ISAAC_PARAM(std::string, first_lhs_frame);
  // Name of the right hand side frame of the first pose
  ISAAC_PARAM(std::string, first_rhs_frame);
  // Name of the left hand side frame of the second pose
  ISAAC_PARAM(std::string, second_lhs_frame);
  // Name of the right hand side frame of the second pose
  ISAAC_PARAM(std::string, second_rhs_frame);

  // This codelet reports success when the difference between two poses is less than this threshold
  // in position and angle.
  ISAAC_PARAM(Vector2d, threshold, Vector2d(0.5, DegToRad(15.0)));
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::Pose2Comparer);
