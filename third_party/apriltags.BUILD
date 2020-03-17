"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
# Description:
#   Apriltag is a library to detect fiducials

cc_library(
    name = "apriltags",
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = select({
        "@com_nvidia_isaac//engine/build:platform_x86_64": ["apriltags_x86_64"],
        "@com_nvidia_isaac//engine/build:platform_jetpack43": ["apriltags_aarch64_jetpack42"],
    }),
)

cc_library(
    name = "apriltags_x86_64",
    srcs = [
        "lib_x64/libapril_tagging.a",
    ],
    hdrs = [
        "nvapriltags/nvAprilTags.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@com_nvidia_isaac//third_party:cuda"],
)

cc_library(
    name = "apriltags_aarch64_jetpack42",
    srcs = [
        "lib_aarch64_jetpack42/libapril_tagging.a",
    ],
    hdrs = [
        "nvapriltags/nvAprilTags.h",
    ],
    visibility = ["//visibility:public"],
    deps = ["@com_nvidia_isaac//third_party:cuda"],
)
