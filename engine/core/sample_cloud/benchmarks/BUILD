"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library", "isaac_pkg")
load("//engine/build:cc_cuda_library.bzl", "cc_cuda_library")

cc_cuda_library(
    name = "cuda_memory_access",
    srcs = [
        "memory_access.cu.cpp",
    ],
    hdrs = [
        "memory_access.cu.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = ["//third_party:cudart"],
)

isaac_cc_binary(
    name = "memory_access",
    srcs = ["memory_access.cpp"],
    deps = [
        ":cuda_memory_access",
        "//engine/core/sample_cloud",
        "//engine/gems/sample_cloud",
        "@benchmark",
    ],
)

isaac_pkg(
    name = "samplecloud_benchmark-pkg",
    srcs = [
        "memory_access"
    ],
)
