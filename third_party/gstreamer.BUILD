"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
# GStreamer pipeline-based multimedia framework with various media processing systems.
# Support for GStreamer functionality and application interfaces included.
cc_library(
    name = "gstreamer",
    hdrs = glob([
        "gst/**/*.h",
        "gst/*.h",
    ]),
    linkopts = [
        "-lgstreamer-1.0",
        "-lgstapp-1.0",
    ],
    strip_include_prefix = "",
    visibility = ["//visibility:public"],
)
