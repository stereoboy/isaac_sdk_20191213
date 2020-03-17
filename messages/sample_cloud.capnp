#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0xecaabaf6f6c1fdea;

using import "element_type.capnp".ElementType;

# a collection of data samples where each sample has N data channels of the
# the same data type, e.g., a Set of 3D points or a RGB color map for a set of points
struct SampleCloudProto {
  # Memory storage layout for the sample cloud data
  enum StorageOrder {
    # data is interleaved, e.g., rgbrgbrgb...
    interleaved @0;
    # data is stored in planes, e.g., rrrr...gggg...bbbb...
    planar @1;
  }

  # The type of data stored in the sample cloud
  elementType @0: ElementType;
  # the number of data channels per sample
  channels @1: UInt16;
  # the number of data samples
  sampleCount @2: UInt64;
  # specifies the memory layout used for storage.
  storageOrder @3: StorageOrder;
  # Index of buffer which contains the sample cloud data
  dataBufferIndex @4: UInt16;
}

# A list of sample clouds, For example one might split geometric and color data
# for a color coded point cloud into two sample clouds rather than a combined data set.
struct SampleCloudListProto {
  samples @0: List(SampleCloudProto);
}
