"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
# GLib low-level libraries for data structure handling for C, portability wrappers, execution
# loops, and interfaces support.
cc_library(
    name = "glib",
    hdrs = glob([
        "**/*.h",
    ]),
    linkopts = [
        "-lglib-2.0",
        "-lgobject-2.0",
    ],
    strip_include_prefix = "",
    visibility = ["//visibility:public"],
)
