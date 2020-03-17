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
    name = "sample_cloud",
    hdrs = [
        "sample_cloud.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/array",
        "//engine/core/buffers",
        "//engine/core/buffers:buffer",
        "//engine/core/math",
        "//engine/core/tensor",
        "//engine/gems/math",
        "@org_tuxfamily_eigen//:eigen"
    ],
)
