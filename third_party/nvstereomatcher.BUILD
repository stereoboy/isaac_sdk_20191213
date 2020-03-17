"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_library(
    name = "nvstereomatcher_x86_64",
    srcs = [
        "lib_x64/libnvstereomatcher.a",
    ],
    hdrs = [
        "nvstereomatcher/nvstereomatcher.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@com_nvidia_isaac//third_party:cuda"],
)

cc_library(
    name = "nvstereomatcher_aarch64_jetpack42",
    srcs = [
        "lib_aarch64_jetpack42/libnvstereomatcher.a",
    ],
    hdrs = [
        "nvstereomatcher/nvstereomatcher.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@com_nvidia_isaac//third_party:cuda"],
)
