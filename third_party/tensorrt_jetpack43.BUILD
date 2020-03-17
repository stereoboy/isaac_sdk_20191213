"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""
# tensorrt is necessary for running inference in the nvstereonet codelet
# It comes preinstalled with jetpack 4.3

cc_library(
    name = "tensorrt_aarch64_jetpack43",
    srcs = [
        "usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so",
        "usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so.6",
        "usr/lib/aarch64-linux-gnu/libnvcaffe_parser.so.6.0.1",
        "usr/lib/aarch64-linux-gnu/libnvinfer.so",
        "usr/lib/aarch64-linux-gnu/libnvinfer.so.6",
        "usr/lib/aarch64-linux-gnu/libnvinfer.so.6.0.1",
        "usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so",
        "usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.6",
        "usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so.6.0.1",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser.so",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser.so.6",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser.so.6.0.1",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser_runtime.so",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser_runtime.so.6",
        "usr/lib/aarch64-linux-gnu/libnvonnxparser_runtime.so.6.0.1",
        "usr/lib/aarch64-linux-gnu/libnvparsers.so",
        "usr/lib/aarch64-linux-gnu/libnvparsers.so.6",
        "usr/lib/aarch64-linux-gnu/libnvparsers.so.6.0.1",
    ],
    hdrs = [
        "usr/include/aarch64-linux-gnu/NvCaffeParser.h",
        "usr/include/aarch64-linux-gnu/NvInfer.h",
        "usr/include/aarch64-linux-gnu/NvInferPlugin.h",
        "usr/include/aarch64-linux-gnu/NvInferPluginUtils.h",
        "usr/include/aarch64-linux-gnu/NvInferRuntime.h",
        "usr/include/aarch64-linux-gnu/NvInferRuntimeCommon.h",
        "usr/include/aarch64-linux-gnu/NvInferVersion.h",
        "usr/include/aarch64-linux-gnu/NvOnnxConfig.h",
        "usr/include/aarch64-linux-gnu/NvOnnxParser.h",
        "usr/include/aarch64-linux-gnu/NvOnnxParserRuntime.h",
        "usr/include/aarch64-linux-gnu/NvUffParser.h",
    ],
    strip_include_prefix = "usr/include/aarch64-linux-gnu",
    visibility = ["//visibility:public"],
    deps = [
        "@cuda_aarch64_jetpack43//:cudnn",
    ],
)
