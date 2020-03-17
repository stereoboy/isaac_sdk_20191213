"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library")

isaac_cc_library(
    name = "common",
    hdrs = ["domains.hpp"],
    deps = ["//engine/gems/state"],
)

cc_test(
    name = "io",
    size = "small",
    srcs = ["io.cpp"],
    deps = [
        ":common",
        "//engine/gems/state",
        "//engine/gems/state:io",
        "@gtest//:main",
    ],
)

cc_test(
    name = "state",
    size = "small",
    srcs = ["state.cpp"],
    deps = [
        ":common",
        "//engine/gems/math:test_utils",
        "//engine/gems/state",
        "@gtest//:main",
    ],
)
