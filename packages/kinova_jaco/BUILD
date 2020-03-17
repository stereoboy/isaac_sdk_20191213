"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_module")

isaac_cc_module(
    name = "kinova_jaco",
    srcs = [
        "KinovaJaco.cpp",
    ],
    hdrs = [
        "KinovaJaco.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/gems/state:io",
        "//messages/state:kinova_jaco",
        "//packages/kinova_jaco/gems:kinova_jaco",
    ],
)
