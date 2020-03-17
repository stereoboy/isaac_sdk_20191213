/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Plan2Generator.hpp"

#include "messages/math.hpp"

namespace isaac {
namespace message_generators {

void Plan2Generator::start() {
  tickPeriodically();
}

void Plan2Generator::tick() {
  auto proto = tx_plan().initProto();
  auto poses_proto = proto.initPoses(get_count());
  const Vector2d step = get_step();
  Pose2d start = Pose2d::Identity();
  for (size_t i = 0; i < poses_proto.size(); i++) {
    ToProto(start, poses_proto[i]);
    start.translation += step;
  }
  proto.setPlanFrame("robot");
  tx_plan().publish();
}

}  // namespace message_generators
}  // namespace isaac
