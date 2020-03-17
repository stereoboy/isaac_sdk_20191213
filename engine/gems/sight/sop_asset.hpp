/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "engine/core/image/image.hpp"
#include "engine/gems/image/io.hpp"
#include "engine/gems/serialization/base64.hpp"
#include "engine/gems/serialization/json.hpp"

namespace isaac {
namespace sight {

// Sight Operation Asset.
// TODO: Right it now it contains only a name to an asset, later we need to add information about
// the asset as well.
class SopAsset {
 public:
  // Delete copy constructor
  SopAsset(const SopAsset&) = delete;
  SopAsset(SopAsset&&) = default;

  static SopAsset FromName(std::string name) {
    SopAsset asset;
    asset.json_["t"] = "asset";
    asset.json_["n"] = std::move(name);
    return asset;
  }

  // Returns name of the SopAsset.
  std::string name() const {
    return json_["n"].get<std::string>();
  }

 private:
  friend const Json& ToJson(const SopAsset&);
  friend Json ToJson(SopAsset&&);

  // Private to allow construction from the static function
  SopAsset() = default;

  Json json_;
};

// Returns the json of a SopAsset
const Json& ToJson(const SopAsset&);
Json ToJson(SopAsset&&);

}  // namespace sight
}  // namespace isaac
