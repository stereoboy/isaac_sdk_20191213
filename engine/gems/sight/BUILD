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
    name = "sight",
    srcs = [
        "named_sop.cpp",
        "sight.cpp",
        "sop_asset.cpp",
        "sop_image.cpp",
        "sop_point_cloud.cpp",
        "sop_style.cpp",
        "sop_text.cpp",
        "sop_transform.cpp",
        "utils.cpp",
    ],
    hdrs = [
        "named_sop.hpp",
        "serialize.hpp",
        "sight.hpp",
        "sight_interface.hpp",
        "sop.hpp",
        "sop_asset.hpp",
        "sop_image.hpp",
        "sop_point_cloud.hpp",
        "sop_style.hpp",
        "sop_text.hpp",
        "sop_transform.hpp",
        "utils.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/sample_cloud",
        "//engine/gems/geometry:pinhole",
        "//engine/gems/geometry:types",
        "//engine/gems/image",
        "//engine/gems/math",
        "//engine/gems/serialization:base64",
        "//engine/gems/serialization:json",
    ],
)

isaac_cc_library(
    name = "kinematics_json",
    srcs = [
        "kinematics_json.cpp",
    ],
    hdrs = [
        "kinematics_json.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/gems/image",
        "//engine/gems/serialization:json",
        "//engine/gems/sight",
        "//third_party/nlohmann",
    ],
)
