"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//third_party:cuda.bzl", "cuda_deps")

cuda_deps()

cc_library(
    name = "tensorrt",
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": ["@tensorrt_x86_64"],
        "//engine/build:platform_jetpack43": ["@tensorrt_aarch64_jetpack43"],
    }) + [
        "cublas",
    ],
)

cc_library(
    name = "libtensorflow",
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": ["@libtensorflow_x86_64"],
        "//engine/build:platform_jetpack43": ["@libtensorflow_aarch64_jetpack43"],
    }) + [
        "//third_party:cublas",
        "//third_party:cudnn",
        "//third_party:cufft",
        "//third_party:curand",
        "//third_party:cusolver",
    ],
)

cc_library(
    name = "libtorch",
    linkstatic = True,
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": ["@libtorch_x86_64"],
        "//engine/build:platform_jetpack43": ["@libtorch_aarch64_jetpack42"],
    }) + [
        "//third_party:cublas",
        "//third_party:cuda",
        "//third_party:cudnn",
        "//third_party:cufft",
        "//third_party:curand",
        "//third_party:cusparse",
        "//third_party:nvrtc",
    ],
)

cc_library(
    name = "ros",
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": ["@isaac_ros_bridge_x86_64"],
        "//engine/build:platform_jetpack43": ["@isaac_ros_bridge_aarch64_xavier"],
    }),
)

cc_library(
    name = "opencv",
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": ["@opencv_x86_64"],
        "//engine/build:platform_jetpack43": ["@opencv_aarch64_jetpack42"],
    }),
)

cc_library(
    name = "libargus",
    visibility = ["//visibility:public"],
    deps = select({
        "//engine/build:platform_x86_64": [],
        "//engine/build:platform_jetpack43": ["@libargus_aarch64_nano"],
    }),
)

cc_library(
    name = "zed",
    visibility = ["//visibility:public"],
    deps = [":cuda"] + select({
        "//engine/build:platform_x86_64": [
            "@zed_x86_64",
            "@libpng12_x86_64//:libpng12_x86_64",
        ],
        "//engine/build:platform_jetpack43": [
            "@zed_aarch64_jetpack42",
        ],
    }),
)
