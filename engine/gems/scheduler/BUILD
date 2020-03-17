"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library", "isaac_doc_dep")

isaac_doc_dep(
    name = "doc",
    srcs = [
        "doc/scheduler.rst",
    ],
    visibility = ["//visibility:public"],
)

isaac_cc_library(
    name = "scheduler",
    srcs = glob(["*.cpp"]),
    hdrs = glob(["*.hpp"]),
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/gems/algorithm:string_utils",
        "//engine/gems/math",
        "//engine/gems/uuid",
        "//third_party:cuda",
    ],
)
