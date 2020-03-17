#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0x9ef78e7aa95c4977;

using import "math.capnp".Vector3dProto;

# A message used to send actuator commands to hardware (or simulated hardware)
struct ActuatorGroupProto {
  # The semantic meaning of the value
  # Represents angular/linear values depending on the joint type
  enum Semantics {
    position @0;
    speed @1;
    acceleration @2;
    force @3;
  }
  # A commanded property
  struct Value {
    semantics @0: Semantics;
    # the actual control or state value
    scalar @1: Float32;
  }
  # Just holds a list of commands per actuator
  struct CommandList {
    commands @0: List(Value);
  }
  # Desired commands per actuators
  actuators @0: List(CommandList);
  # (optional) names of actuators
  names @1: List(Text);
}
