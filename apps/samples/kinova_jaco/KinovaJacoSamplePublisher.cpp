/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "KinovaJacoSamplePublisher.hpp"

#include "messages/messages.hpp"
#include "messages/state/kinova_jaco.hpp"

namespace isaac {
namespace kinova_jaco {

void KinovaJacoSamplePublisher::start() {
  tickOnMessage(rx_cartesian_pose());
}

void KinovaJacoSamplePublisher::tick() {
  // Read cartesian pose from message
  messages::JacoCartesianPose cartesian_pose;
  FromProto(rx_cartesian_pose().getProto(), rx_cartesian_pose().buffers(), cartesian_pose);

  // Display position in Sight
  show("px", cartesian_pose.px());
  show("py", cartesian_pose.py());
  show("pz", cartesian_pose.pz());

  // Display orientation in Sight
  show("qw", cartesian_pose.qw());
  show("qx", cartesian_pose.qx());
  show("qy", cartesian_pose.qy());
  show("qz", cartesian_pose.qz());
}

}  // namespace kinova_jaco
}  // namespace isaac
