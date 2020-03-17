/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Pose2Comparer.hpp"

#include <string>

#include "messages/math.hpp"

namespace isaac {
namespace alice {

void Pose2Comparer::start() {
  tickPeriodically();
}

void Pose2Comparer::tick() {
  // Read poses
  const auto maybe_pose1 =
      node()->pose().tryGetPose2XY(get_first_lhs_frame(), get_first_rhs_frame(), getTickTime());
  const auto maybe_pose2 =
      node()->pose().tryGetPose2XY(get_second_lhs_frame(), get_second_rhs_frame(), getTickTime());
  if (!maybe_pose1 || !maybe_pose2) {
    return;
  }

  // Read and check parameters
  const Vector2d threshold = get_threshold();
  if (threshold[0] < 0.0) {
    reportFailure("Negative position threshold");
    return;
  }
  if (threshold[1] < 0.0) {
    reportFailure("Negative angle threshold");
    return;
  }

  // Check difference
  const Pose2d difference = maybe_pose1->inverse() * *maybe_pose2;
  if (difference.translation.norm() < threshold[0] &&
      std::abs(difference.rotation.angle()) < threshold[1]) {
    reportSuccess();
  }
}

}  // namespace alice
}  // namespace isaac
