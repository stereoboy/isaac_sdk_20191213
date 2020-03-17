/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/components/PoseTree.hpp"

#include <memory>
#include "engine/alice/component.hpp"
#include "engine/gems/math/test_utils.hpp"

#include "gtest/gtest.h"

TEST(PoseTree, isConstructible) {
  // test if PoseTree is constructible

  auto poseTree = std::make_unique<isaac::alice::PoseTree>();
  EXPECT_TRUE(poseTree);
}

TEST(PoseTree, callbackCalledOnSet) {
  // test if a registered callback is called upon set

  isaac::alice::PoseTree poseTree;               // object under test
  isaac::alice::Component* component = nullptr;  // stand in for a real component

  bool callbackFlag = false;  // callback will set to true when called
  isaac::alice::PoseTree::UpdateFunction callback =
      [&callbackFlag](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
                      const isaac::Pose3d& lhs_T_rhs) {
        callbackFlag = true;  // set callback flag so we know callback was called successfully
      };

  poseTree.registerForUpdates(component, callback);
  poseTree.set("a", "b", isaac::Pose3d{}, 0.0);

  EXPECT_TRUE(callbackFlag);
}

TEST(PoseTree, callbackStopsOnDeregister) {
  // test if a deregistered callback is NOT called upon set

  isaac::alice::PoseTree poseTree;               // object under test
  isaac::alice::Component* component = nullptr;  // stand in for a real component

  isaac::alice::PoseTree::UpdateFunction callback =
      [](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
         const isaac::Pose3d& lhs_T_rhs) { FAIL() << "callback should not have been called"; };

  poseTree.registerForUpdates(component, callback);
  poseTree.deregisterForUpdates(component);
  poseTree.set("a", "b", isaac::Pose3d{}, 0.0);
}

TEST(PoseTree, callbackReceivesCorrectParameters) {
  // test if calback receives correct parameters

  isaac::alice::PoseTree poseTree;               // object under test
  isaac::alice::Component* component = nullptr;  // stand in for a real component

  const isaac::Pose3d pose3d = isaac::Pose3d::Translation(1.0, 2.0, 3.0);  // used as a test value

  isaac::alice::PoseTree::UpdateFunction callback = [&pose3d](const isaac::Uuid& lhs,
                                                              const isaac::Uuid& rhs, double stamp,
                                                              const isaac::Pose3d& lhs_T_rhs) {
    EXPECT_EQ(lhs.str(), "a");
    EXPECT_EQ(rhs.str(), "b");
    EXPECT_NEAR(stamp, 1.0, 0.00001);
    ISAAC_EXPECT_VEC_NEAR(lhs_T_rhs.translation, pose3d.translation, 0.001);
  };

  poseTree.registerForUpdates(component, callback);
  poseTree.set("a", "b", pose3d, 1.0);
}

TEST(PoseTree, multipleCallbacks) {
  // test if multiple callbacks are registered, both receive the callback

  isaac::alice::PoseTree poseTree;     // object under test
  isaac::alice::Component componentA;  // stand in for a real component
  isaac::alice::Component componentB;  // stand in for a real component

  bool callbackFlagA = false;  // callbackA will set to true when called
  isaac::alice::PoseTree::UpdateFunction callbackA =
      [&callbackFlagA](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
                       const isaac::Pose3d& lhs_T_rhs) {
        callbackFlagA = true;  // set callback flag so we know callback was called successfully
      };

  bool callbackFlagB = false;  // callbackB will set to true when called
  isaac::alice::PoseTree::UpdateFunction callbackB =
      [&callbackFlagB](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
                       const isaac::Pose3d& lhs_T_rhs) {
        callbackFlagB = true;  // set callback flag so we know callback was called successfully
      };

  poseTree.registerForUpdates(&componentA, callbackA);
  poseTree.registerForUpdates(&componentB, callbackB);
  poseTree.set("a", "b", isaac::Pose3d{}, 0.0);

  EXPECT_TRUE(callbackFlagA);
  EXPECT_TRUE(callbackFlagB);
}

TEST(PoseTree, multipleCallbacksWithDeregister) {
  // test if multiple callbacks are registered, then one deregisters. Only one should receive the
  // callback

  isaac::alice::PoseTree poseTree;     // object under test
  isaac::alice::Component componentA;  // stand in for a real component
  isaac::alice::Component componentB;  // stand in for a real component

  bool callbackFlagA = false;  // callbackA will set to true when called
  isaac::alice::PoseTree::UpdateFunction callbackA =
      [&callbackFlagA](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
                       const isaac::Pose3d& lhs_T_rhs) {
        callbackFlagA = true;  // set callback flag so we know callback was called successfully
      };

  bool callbackFlagB = false;  // callbackB will set to true when called
  isaac::alice::PoseTree::UpdateFunction callbackB =
      [&callbackFlagB](const isaac::Uuid& lhs, const isaac::Uuid& rhs, double stamp,
                       const isaac::Pose3d& lhs_T_rhs) {
        callbackFlagB = true;  // set callback flag so we know callback was called successfully
      };

  poseTree.registerForUpdates(&componentA, callbackA);
  poseTree.registerForUpdates(&componentB, callbackB);
  poseTree.deregisterForUpdates(&componentA);
  poseTree.set("a", "b", isaac::Pose3d{}, 0.0);

  EXPECT_FALSE(callbackFlagA);  // callbackA was deregistered, so should not be set
  EXPECT_TRUE(callbackFlagB);   // callbackB remained registered, so should be set
}
