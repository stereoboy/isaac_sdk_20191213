"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_subgraph")

isaac_subgraph(
    name = "yolo_detection_subgraph",
    modules = [
        "ml",
        "viewers",
        "yolo",
    ],
    subgraph = "yolo_detection.subgraph.json",
    visibility = ["//visibility:public"],
)
