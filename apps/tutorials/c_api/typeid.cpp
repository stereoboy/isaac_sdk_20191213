/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <cstdio>

#include "capnp/schema.h"

#include "messages/messages.hpp"

#define PRINT_MACRO(PROTO) \
  std::printf("Proto Id for " #PROTO " is %lu\n", ::capnp::typeId<PROTO>());

int main(int argc, char** argv) {
  PRINT_MACRO(ActorGroupProto)
  PRINT_MACRO(CollisionProto)
  PRINT_MACRO(ColorCameraProto)
  PRINT_MACRO(DepthCameraProto)
  PRINT_MACRO(SegmentationCameraProto)
  PRINT_MACRO(FlatscanProto)
  PRINT_MACRO(JsonProto)
  PRINT_MACRO(Plan2Proto)
  PRINT_MACRO(PoseTreeEdgeProto)
  PRINT_MACRO(RangeScanProto)
  PRINT_MACRO(RigidBody3GroupProto)
  PRINT_MACRO(StateProto)
}
