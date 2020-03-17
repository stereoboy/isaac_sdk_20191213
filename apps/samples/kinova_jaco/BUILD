"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_cc_module")

isaac_cc_module(
    name = "sample_controller",
    srcs = [
        "KinovaJacoSampleController.cpp",
    ],
    hdrs = [
        "KinovaJacoSampleController.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/gems/state:io",
        "//messages/state:kinova_jaco",
    ],
)

isaac_cc_module(
    name = "sample_publisher",
    srcs = [
        "KinovaJacoSamplePublisher.cpp",
    ],
    hdrs = [
        "KinovaJacoSamplePublisher.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/gems/state:io",
        "//messages/state:kinova_jaco",
    ],
)

isaac_app(
    name = "kinova_jaco",
    modules = [
        "kinova_jaco",
        "//apps/samples/kinova_jaco:sample_controller",
        "//apps/samples/kinova_jaco:sample_publisher",
    ],
)
