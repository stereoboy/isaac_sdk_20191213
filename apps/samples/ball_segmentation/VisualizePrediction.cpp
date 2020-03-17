/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "VisualizePrediction.hpp"

#include "engine/core/image/image.hpp"
#include "engine/gems/image/color.hpp"
#include "engine/gems/image/conversions.hpp"
#include "engine/gems/image/io.hpp"
#include "engine/gems/image/utils.hpp"
#include "messages/tensor.hpp"

namespace isaac {
namespace ball_segmentation {

void VisualizePrediction::start() {
  tickOnMessage(rx_segmentation_prediction());
  synchronize(rx_image_tensor(), rx_segmentation_prediction());
}

void VisualizePrediction::tick() {
  // read in image/segmentation prediction tensors
  TensorConstView3f prediction;
  TensorConstView3f image_tensor;
  auto image_tensor_reader = rx_image_tensor().getProto().getTensors()[0];
  auto segmentation_prediction_reader = rx_segmentation_prediction().getProto();
  bool ok = FromProto(image_tensor_reader, rx_image_tensor().buffers(), image_tensor);
  if (!ok) {
    return;
  }
  ok = FromProto(segmentation_prediction_reader.getPrediction(),
                 rx_segmentation_prediction().buffers(), prediction);
  if (!ok) {
    return;
  }

  const size_t tensor_rows = prediction.dimensions()[0];
  const size_t tensor_cols = prediction.dimensions()[1];
  const size_t image_tensor_rows = image_tensor.dimensions()[0];
  const size_t image_tensor_cols = image_tensor.dimensions()[1];

  ASSERT(tensor_rows == image_tensor_rows && tensor_cols == image_tensor_cols,
         "tensor sizes mismatch");

  // generate probability image
  probability_image_.resize(tensor_rows, tensor_cols);
  for (size_t row = 0; row < tensor_rows; row++) {
    for (size_t col = 0; col < tensor_cols; col++) {
      probability_image_(row, col) = prediction(row, col, 0);
    }
  }

  // generate segmentation image and join them side by side
  NormalizedTensorToImage(image_tensor, ImageToTensorNormalization::kPositiveNegative,
                          left_image_);
  Colorize(probability_image_, BlackWhiteColorGradient(), 0.0f, 1.0f, right_image_);
  JoinTwoImagesSideBySide(left_image_, right_image_, joint_image_);

  // show raw image and prediction in one 3ub image.
  show("image_with_prediction", joint_image_);
}

}  // namespace ball_segmentation
}  // namespace isaac
