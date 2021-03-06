/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "TorchInferenceTestSendInput.hpp"

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include "messages/tensor.hpp"

namespace isaac {

void TorchInferenceTestSendInput::start() {
  tickPeriodically();
}

void TorchInferenceTestSendInput::tick() {
  auto input_tensors = tx_test_input().initProto().initTensors(1);
  Tensor2f tensor_out(1, 1);
  tensor_out(0, 0) = get_input_value();
  LOG_INFO("Torch Inference test input: %f", get_input_value());

  ToProto(std::move(tensor_out), input_tensors[0], tx_test_input().buffers());
  tx_test_input().publish();
}

}  // namespace isaac
