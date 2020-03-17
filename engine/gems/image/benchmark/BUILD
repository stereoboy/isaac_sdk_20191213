"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library", "isaac_pkg")

isaac_cc_binary(
    name = "aliasing",
    srcs = ["aliasing.cpp"],
    deps = [
        "@benchmark",
    ],
)

isaac_cc_binary(
    name = "compression",
    srcs = ["benchmark.cpp"],
    data = ["//engine/gems/image/data"],
    deps = [
        "//engine/gems/image",
        "@benchmark",
    ],
)

isaac_cc_binary(
    name = "normalize",
    srcs = ["normalize.cpp"],
    data = ["//engine/gems/image/data"],
    deps = [
        "//engine/gems/image",
        "//engine/gems/image:cuda_conversions",
        "@benchmark",
    ],
)

isaac_pkg(
    name = "image_benchmark-pkg",
    srcs = [
        "aliasing",
        "compression",
        "normalize",
    ],
)
