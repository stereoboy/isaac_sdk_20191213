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
#include "engine/core/image/image.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "messages/messages.hpp"

namespace isaac {
namespace ball_segmentation {

// Visualize the probability image with the original image side-by-side
class VisualizePrediction : public alice::Codelet {
 public:
  void start() override;
  void tick() override;

  // raw rgb (down-sized) image in tensor form
  ISAAC_PROTO_RX(TensorListProto, image_tensor)
  // segmentation prediction in tensor form
  ISAAC_PROTO_RX(SegmentationPredictionProto, segmentation_prediction)

 private:
  // pre-allocated buffers
  Image3ub left_image_;
  Image3ub right_image_;
  Image1f probability_image_;
  Image3ub joint_image_;
};

}  // namespace ball_segmentation
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::ball_segmentation::VisualizePrediction);
