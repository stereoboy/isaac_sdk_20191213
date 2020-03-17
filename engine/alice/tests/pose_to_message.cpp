/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/components/PoseToMessage.hpp"
#include "engine/gems/math/test_utils.hpp"
#include "gtest/gtest.h"
#include "messages/alice.capnp.h"
#include "messages/math.hpp"

namespace isaac {
namespace alice {

namespace {
constexpr char kLhsFrame[] = "lhs";
constexpr char kRhsFrame[] = "rhs";
}  // namespace

class MyPoseTest : public Codelet {
 public:
  void start() override {
    const bool ok = node()->pose().set(kLhsFrame, kRhsFrame, pose_, getTickTime());
    EXPECT_TRUE(ok);
    tickOnMessage(rx_pose());
  }
  void tick() override {
    rx_pose().processAllNewMessages([&](auto proto, int64_t pubtime, int64_t acqtime) {
      EXPECT_STREQ(proto.getLhs().cStr(), kLhsFrame);
      EXPECT_STREQ(proto.getRhs().cStr(), kRhsFrame);
      ISAAC_EXPECT_POSE_NEAR(FromProto(proto.getPose()).toPose2XY(), pose_, 1e-12);
    });
  }
  void stop() override {}
  void set_pose(Pose2d pose) { pose_ = std::move(pose); }

  ISAAC_PROTO_RX(PoseTreeEdgeProto, pose);

 private:
  Pose2d pose_;
};

TEST(PoseToMessage, ExistingPose) {
  const Pose2d pose{SO2d::FromAngle(DegToRad(60.0)), Vector2d{3.0, -2.1}};

  Application app;
  Node* node = app.createNode("test");
  node->addComponent<MessageLedger>();
  auto* test_component = node->addComponent<MyPoseTest>();
  test_component->set_pose(pose);
  auto* pose_to_message = node->addComponent<PoseToMessage>();
  pose_to_message->async_set_tick_period("100ms");
  pose_to_message->async_set_lhs_frame(kLhsFrame);
  pose_to_message->async_set_rhs_frame(kRhsFrame);
  Connect(pose_to_message->tx_pose(), test_component->rx_pose());
  app.startWaitStop(0.25);

  EXPECT_GT(test_component->getTickCount(), 1);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyPoseTest);
