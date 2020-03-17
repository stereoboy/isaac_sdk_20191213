"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_library")

isaac_cc_library(
    name = "math",
    hdrs = glob(
        ["*.hpp"],
        exclude = [
            "catmull_rom.hpp",
            "pose_utils.hpp",
        ],
    ),
    visibility = ["//visibility:public"],
    deps = [
        "catmull_rom",
        "icosphere_sampling",
        "//engine/core",
        "//engine/core/math",
        "//third_party/half",
        "@org_tuxfamily_eigen//:eigen",
    ],
)

isaac_cc_library(
    name = "catmull_rom",
    hdrs = ["catmull_rom.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/math",
    ],
)

isaac_cc_library(
    name = "icosphere_sampling",
    hdrs = ["icosphere_sampling.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/math",
    ],
)

isaac_cc_library(
    name = "pose_utils",
    hdrs = ["pose_utils.hpp"],
    visibility = ["//visibility:public"],
    deps = ["//engine/core/math"],
)

isaac_cc_library(
    name = "test_utils",
    hdrs = ["test_utils.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        ":math",
        "@gtest//:main",
    ],
)
