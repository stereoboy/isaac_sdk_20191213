"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "opencv_aarch64_jetpack42",
    srcs = [":so"],
    hdrs = glob([
        "usr/include/**/*.h",
        "usr/include/**/*.hpp",
    ]),
    data = [":so"],
    strip_include_prefix = "usr/include",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "so",
    srcs = glob(["usr/lib/*.so*"]),
)
