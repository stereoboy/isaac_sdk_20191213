/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/alice_codelet.hpp"
#include "engine/alice/status.hpp"

namespace isaac {
namespace alice {
namespace behaviors {

// @experimental
// A simple behavior which always has the same status. It can be used for tests or to build more
// complicated behavior trees.
class ConstantBehavior : public Codelet {
 public:
  void start() override {
    const auto status = get_status();
    if (status != Status::RUNNING) {
      updateStatus(status);
    }
  }

  // The desired status of the behavior. It is set once when the component starts and is not changed
  // afterwards.
  ISAAC_PARAM(Status, status, Status::RUNNING);
};

}  // namespace behaviors
}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::behaviors::ConstantBehavior);
