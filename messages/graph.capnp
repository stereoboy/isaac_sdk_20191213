#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0xaec0fa9c38fbb1ff;

# A general graph
struct GraphProto {
  # The number of nodes in the graph. Indices in the list of edges will
  # always be smaller than this number.
  nodeCount @0: Int32;
  # An edge in the graph connects two nodes.
  struct EdgeProto {
    source @0: Int32;
    target @1: Int32;
  }
  # The edges of the graph.
  edges @1: List(EdgeProto);
}