/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/alice_codelet.hpp"
#include "engine/core/tensor/tensor.hpp"
#include "messages/messages.hpp"

namespace isaac {
namespace message_generators {

// Available output modes for ImageWarp
enum class TensorListGeneratorElementType {
  kFloat32,
  kInt32,
  kInvalid = -1
};

NLOHMANN_JSON_SERIALIZE_ENUM(TensorListGeneratorElementType, {
  { TensorListGeneratorElementType::kInvalid, nullptr },
  { TensorListGeneratorElementType::kFloat32, "float32" },
  { TensorListGeneratorElementType::kInt32, "int32" }
});

// TensorListGenerator creates lists of tensors from nothing.
class TensorListGenerator : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;

  // Produced random list of tensors with the specified dimensions
  ISAAC_PROTO_TX(TensorListProto, sample);

  // Number of tensors per sample
  ISAAC_PARAM(int, num_tensors, 3);
  // Dimensions of the generated rank 3 tensor
  ISAAC_PARAM(Vector3i, dimensions, Vector3i(3, 640, 480));
  // The element type for the tensor
  ISAAC_PARAM(TensorListGeneratorElementType, element_type,
              TensorListGeneratorElementType::kFloat32);

 private:
  Tensor3f buffer_float32_;
  Tensor3i buffer_int32_;
};

}  // namespace message_generators
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::message_generators::TensorListGenerator);
