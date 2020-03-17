"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary")

isaac_cc_binary(
    name = "c_api_viewer",
    srcs = [
        "viewer.c",
    ],
    data = [
        "viewer.app.json",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/alice/c_api",
    ],
)


isaac_cc_binary(
    name = "c_api_depth",
    srcs = [
        "depth.c",
    ],
    data = [
        "depth.app.json",
        "//packages/message_generators:libmessage_generators_module.so",
    ],
    deps = [
        "//engine/alice/c_api",
    ],
)

isaac_cc_binary(
    name = "typeid",
    srcs = [
        "typeid.cpp",
    ],
    deps = [
        "//engine/alice/c_api",
    ],
)
