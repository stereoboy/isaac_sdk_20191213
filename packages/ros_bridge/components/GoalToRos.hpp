/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>

#include "move_base_msgs/MoveBaseAction.h"
#include "packages/ros_bridge/components/ProtoToRosConverter.hpp"

namespace isaac {
namespace ros_bridge {

// This codelet receives goal as message within Isaac application and publishes it to ROS as a
// message. If goal feedback is needed, use similar codelet named "GoalToRosAction" instead.
class GoalToRos : public ProtoToRosConverter<Goal2Proto, move_base_msgs::MoveBaseActionGoal, true> {
 public:
  bool protoToRos(Goal2Proto::Reader reader,
                  move_base_msgs::MoveBaseActionGoal& ros_message) override;

  // Frame of the goal in outgoing message
  ISAAC_PARAM(std::string, goal_frame, "map");
  // Frame of the robot in ROS. Used to stop the robot if needed.
  ISAAC_PARAM(std::string, robot_frame, "base_link");
  // A new message will be published whenever change in goal pose exceeds these thresholds. Values
  // are for Euclidean distance and angle respectively.
  ISAAC_PARAM(Vector2d, new_message_thresholds, Vector2d(1e-3, DegToRad(0.1)));

 private:
  struct Goal {
    Pose2d pose;
    std::string frame;
  };

  // Returns true if goal is different than last_goal_. In this case, we publish a new ROS message.
  bool isNewGoal(const Goal& goal);

  // Last goal sent to ROS
  std::optional<Goal> last_goal_ = std::nullopt;
};

}  // namespace ros_bridge
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::ros_bridge::GoalToRos);
