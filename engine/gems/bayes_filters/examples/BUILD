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
    name = "ekf_sin_exp",
    srcs = ["ekf_sin_exp.cpp"],
    deps = [
        "//engine/gems/bayes_filters",
        "//engine/gems/sight",
    ],
)

isaac_cc_binary(
    name = "ekf_body_1d",
    srcs = ["ekf_body_1d.cpp"],
    deps = [
        "//engine/gems/bayes_filters",
        "//engine/gems/sight",
    ],
)

isaac_cc_binary(
    name = "pf_sin_exp",
    srcs = ["pf_sin_exp.cpp"],
    deps = [
        "//engine/gems/bayes_filters",
        "//engine/gems/sight",
    ],
)
