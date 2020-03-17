/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "sop_point_cloud.hpp"

#include <vector>

#include "engine/gems/serialization/base64.hpp"

namespace isaac {
namespace sight {

SopPointCloud SopPointCloud::Create(SampleCloudConstView3f points, SampleCloudConstView3f colors,
                                    size_t downsample_stride) {
  SopPointCloud sop;
  sop.json_["t"] = "point_cloud";
  if (downsample_stride == 1) {
    sop.json_["points"] = serialization::Base64Encode(
        reinterpret_cast<const uint8_t*>(points.data().pointer().get()), points.size() * 12);
    sop.json_["colors"] = serialization::Base64Encode(
        reinterpret_cast<const uint8_t*>(colors.data().pointer().get()), colors.size() * 12);
  } else {
    SampleCloud3f downsample_points;
    if (points.size() / downsample_stride > 0) {
      downsample_points.resize(points.size() / downsample_stride);
    }
    SampleCloud3f downsample_colors;
    if (colors.size() / downsample_stride > 0) {
      downsample_colors.resize(colors.size() / downsample_stride);
    }
    for (size_t index = 0; index < downsample_points.size(); index++) {
      downsample_points[index] = points[index * downsample_stride];
    }
    for (size_t index = 0; index < downsample_colors.size(); index++) {
      downsample_colors[index] = colors[index * downsample_stride];
    }
    sop.json_["points"] = serialization::Base64Encode(
        reinterpret_cast<const uint8_t*>(downsample_points.data().pointer().get()),
        downsample_points.size() * 12);
    sop.json_["colors"] = serialization::Base64Encode(
        reinterpret_cast<const uint8_t*>(downsample_colors.data().pointer().get()),
        downsample_colors.size() * 12);
  }
  return sop;
}

}  // namespace sight
}  // namespace isaac
