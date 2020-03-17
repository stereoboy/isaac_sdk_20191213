"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_pkg")

isaac_app(
    name = "navsim_server",
    data = [
        "//apps/navsim/multirobot/scenarios",
        "//engine/alice/c_api:libisaac_c_api.so",
    ],
    visibility = ["//visibility:public"],
)

isaac_app(
    name = "navsim_client",
    data = [
        "//apps/navsim/multirobot/configs:client_config",
        "//apps/navsim/multirobot/scenarios",
        "//engine/alice/c_api:libisaac_c_api.so",
    ],
    modules = [
        "imu",
    ],
    visibility = ["//visibility:public"],
)

isaac_pkg(
    name = "navsim-pkg",
    srcs = [
        ":navsim_client",
        ":navsim_server",
    ],
    visibility = ["//visibility:public"],
)

isaac_app(
    name = "mosaic",
    modules = [
        "ml",
        "viewers",
    ],
    visibility = ["//visibility:public"],
)
