"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app")

isaac_app(
    name = "flatsim",
    data = glob(["demo_*.json"]) + [
        "//packages/flatsim/apps:flatsim_subgraph",
        "//packages/navigation/apps:differential_base_navigation_subgraph",
        "//packages/navigation/apps:goal_generators_subgraph",
        "//apps/assets/maps",
    ],
    modules = [
        "flatsim",
        "imu",
        "map",
        "navigation",
        "planner",
        "sight",
        "viewers",
    ],
    script = "flatsim.sh",
)
