/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>

#include "engine/alice/backend/component_backend.hpp"
#include "engine/alice/components/deprecated/Behavior.hpp"

namespace isaac {
namespace alice {

// Backend for Behavior component
class BehaviorBackend : public ComponentBackend<Behavior> {
 public:
  void registerComponent(Behavior* behavior) override;
  void unregisterComponent(Behavior* behavior) override;

 private:
  std::map<Node*, Node*> node_to_parent_map_;
};

}  // namespace alice
}  // namespace isaac
