"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

#Yolo Tensorrt library for optimizing the model and running inference
load("@com_nvidia_isaac//engine/build:cc_cuda_library.bzl", "cc_cuda_library")

yolo_tensorrt_hdrs = [
    "src/yolo.hpp",
    "src/yolo_config.hpp",
    "src/yolov3.hpp",
    "src/yolov3-tiny.hpp",
    "src/trt_utils.hpp",
    "src/yolo_impl.hpp",
    "src/plugin_factory.hpp",
    "src/kernels.cu.hpp",
]

cc_library(
    name = "yolo_tensorrt_lib",
    srcs = [
        "src/plugin_factory.cpp",
        "src/trt_utils.cpp",
        "src/yolo.cpp",
        "src/yolo_impl.cpp",
        "src/yolov3.cpp",
        "src/yolov3-tiny.cpp",
    ],
    hdrs = yolo_tensorrt_hdrs,
    copts = [
        "-Wno-deprecated-declarations",
    ],
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
    deps = [
        "yolo_cuda_kernels",
        "@com_nvidia_isaac//engine/core",
        "@com_nvidia_isaac//engine/gems/serialization:json",
        "@com_nvidia_isaac//third_party:cudnn",
    ],
)

cc_binary(
    name = "libyolo_tensorrt.so",
    copts = [
        "-Wno-deprecated-declarations",
    ],
    linkshared = True,
    visibility = ["//visibility:public"],
    deps = [
        "@yolo_tensorrt_lib",
    ],
)

cc_library(
    name = "yolo_tensorrt_hdrs",
    hdrs = yolo_tensorrt_hdrs,
    strip_include_prefix = "src",
    visibility = ["//visibility:public"],
)

genrule(
    name = "kernels_cu_cpp",
    srcs = ["src/kernels.cu"],
    outs = ["src/kernels.cu.cpp"],
    cmd = "cp $< $@",
)

cc_cuda_library(
    name = "yolo_cuda_kernels",
    srcs = [
        "src/kernels.cu.cpp",
    ],
    hdrs = yolo_tensorrt_hdrs,
    strip_include_prefix = "src",
    deps = [
        "@com_nvidia_isaac//third_party:tensorrt",
    ],
)
