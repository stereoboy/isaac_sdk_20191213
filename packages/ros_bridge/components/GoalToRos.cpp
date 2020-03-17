/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "GoalToRos.hpp"

#include "messages/math.hpp"
#include "tf2/LinearMath/Quaternion.h"

namespace isaac {
namespace ros_bridge {

bool GoalToRos::protoToRos(Goal2Proto::Reader reader,
                           move_base_msgs::MoveBaseActionGoal& ros_message) {
  // Read data from Isaac type
  Goal goal;
  if (reader.getStopRobot()) {
    // Stop the robot by sending current pose as goal
    goal.pose = Pose2d::Identity();
    goal.frame = get_robot_frame();
  } else {
    goal.pose = FromProto(reader.getGoal());
    goal.frame = get_goal_frame();
  }

  // Do not publish if this goal is close enough to the previous one.
  if (!isNewGoal(goal)) {
    return false;
  }
  last_goal_ = goal;

  // Publish goal data to ROS.
  ros_message.header.stamp = ros::Time::now();
  ros_message.goal.target_pose.header.frame_id = goal.frame;
  ros_message.goal.target_pose.header.stamp = ros::Time::now();
  ros_message.goal.target_pose.pose.position.x = goal.pose.translation.x();
  ros_message.goal.target_pose.pose.position.y = goal.pose.translation.y();
  ros_message.goal.target_pose.pose.position.z = 0.0;
  tf2::Quaternion quaternion;
  quaternion.setRPY(0, 0, goal.pose.rotation.angle());
  ros_message.goal.target_pose.pose.orientation.x = quaternion.x();
  ros_message.goal.target_pose.pose.orientation.y = quaternion.y();
  ros_message.goal.target_pose.pose.orientation.z = quaternion.z();
  ros_message.goal.target_pose.pose.orientation.w = quaternion.w();
  return true;
}

bool GoalToRos::isNewGoal(const Goal& goal) {
  if (!last_goal_) {
    // This is the first goal
    return true;
  }

  if (goal.frame != last_goal_->frame) {
    return true;
  }

  const Vector2d thresholds = get_new_message_thresholds();
  if (thresholds[0] < 0.0) {
    reportFailure("Negative position threshold");
    return false;
  }
  if (thresholds[1] < 0.0) {
    reportFailure("Negative angle threshold");
    return false;
  }

  const Pose2d goal_difference = last_goal_->pose.inverse() * goal.pose;
  const bool new_translation = goal_difference.translation.norm() > thresholds[0];
  const bool new_angle = std::abs(goal_difference.rotation.angle()) > thresholds[1];
  return new_translation || new_angle;
}

}  // namespace ros_bridge
}  // namespace isaac
