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
    name = "flatsim_subgraph",
    data = [
        "//packages/navsim/apps:sensor_noiser_subgraph",
    ],
    modules = [
        "flatsim",
        "navsim",
    ],
    subgraph = "2d_differential_base_simulation.subgraph.json",
    visibility = ["//visibility:public"],
)
