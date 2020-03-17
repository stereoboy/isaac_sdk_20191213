/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/alice.hpp"
#include "messages/image.hpp"

namespace isaac {
namespace alice {

// @experimental
// Emits image messages
class ImageTransmitter : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_TX(ImageProto, image);
  ISAAC_PARAM(int, rows, 10);
  ISAAC_PARAM(int, cols, 15);
};

// @experimental
// Receives image messages from ImageTransmitter and checks there integrity
class ImageReceiver : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;
  ISAAC_PROTO_RX(ImageProto, image);
  ISAAC_PARAM(int, rows, 10);
  ISAAC_PARAM(int, cols, 15);
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ImageTransmitter);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ImageReceiver);
