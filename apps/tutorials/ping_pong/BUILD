"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_cc_module")

isaac_cc_module(
    name = "ping_components",
    srcs = [
        "Ping.cpp",
        "Pong.cpp",
    ],
    hdrs = [
        "Ping.hpp",
        "Pong.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [],
)

isaac_app(
    name = "ping_pong",
    modules = [
        "//apps/tutorials/ping_pong:ping_components",
    ],
)
