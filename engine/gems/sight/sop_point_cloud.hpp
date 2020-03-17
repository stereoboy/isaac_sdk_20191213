/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <utility>

#include "engine/core/math/types.hpp"
#include "engine/core/sample_cloud/sample_cloud.hpp"
#include "engine/gems/serialization/json.hpp"

namespace isaac {
namespace sight {

// Sight shop operation for point clouds.
class SopPointCloud {
 public:
  // Delete copy constructor
  SopPointCloud(const SopPointCloud&) = delete;
  SopPointCloud(SopPointCloud&&) = default;

  // Creates a show operation for point clouds
  // This function assumes that the data is 3 dimensionsal and consists of floats.
  // down_sample_stride can be used to render less data than provided, but
  // will force an aditional copy at this time.
  static SopPointCloud Create(
      SampleCloudConstView3f points,
      SampleCloudConstView3f colors = SampleCloudConstView3f(),
      size_t downsample_stride = 1);

 private:
  friend const Json& ToJson(const SopPointCloud&);
  friend Json ToJson(SopPointCloud&&);

  // Private to allow construction from the static function
  SopPointCloud() = default;

  Json json_;
};

// Returns the json of a SopPointCloud
inline const Json& ToJson(const SopPointCloud& sop) {
  return sop.json_;
}

inline Json ToJson(SopPointCloud&& sop) {
  return std::move(sop.json_);
}

}  // namespace sight
}  // namespace isaac
