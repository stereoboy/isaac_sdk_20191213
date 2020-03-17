"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library", "isaac_pkg")

isaac_cc_library(
    name = "c_api",
    srcs = [
        "application_c_api.cpp",
        "application_c_api.hpp",
        "isaac_c_api.cpp",
        "isaac_c_api_error.cpp",
        "message_c_api.cpp",
        "message_c_api.hpp",
    ],
    hdrs = [
        "isaac_c_api.h",
        "isaac_c_api_types.h",
        "isaac_c_api_error.h",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/alice",
        "//engine/alice/tools:websight",
        "//engine/core",
        "//messages",
    ],
)

isaac_cc_binary(
    name = "libisaac_c_api.so",
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = ["c_api"],
)

filegroup(
    name = "public_headers",
    data = [
        "isaac_c_api.h",
        "isaac_c_api_error.h",
        "isaac_c_api_types.h",
    ],
)

isaac_pkg(
    name = "isaac_engine_c_api-pkg",
    srcs = [
        "libisaac_c_api.so",
        "public_headers",
    ],
)
