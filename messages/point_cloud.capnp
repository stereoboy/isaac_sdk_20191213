#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0x8846db30e403e12a;

using import "sample_cloud.capnp".SampleCloudProto;

# A message about a cloud of points.
struct PointCloudProto {
  # The positions of the points
  positions @0:  SampleCloudProto;
  # Optional data about the normals at each point. Must match the number of positions.
  normals @1: SampleCloudProto;
  # The colors at each position. This is optional, but if present must have the same length as `positions`.
  colors @2: SampleCloudProto;
  # The intensity at each position, used for example by LIDAR sensors. This is optional, but if present must
  # have the same length as `positions`.
  intensities @3: SampleCloudProto;
}
