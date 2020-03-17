"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

exports_files(["LICENSE.TXT"])

cc_library(
    name = "zed_aarch64_jetpack42",
    srcs = [
        "aarch64/lib/libsl_zed.so",
        "aarch64/lib/libsl_input.so",
        "aarch64/lib/libsl_core.so",
    ],
    hdrs = glob(["include/**/*.hpp"]),
    includes = ["include"],
    visibility = ["//visibility:public"],
    deps = [
        "@com_nvidia_isaac//third_party:cuda",
        "@com_nvidia_isaac//third_party:nppial",
        "@com_nvidia_isaac//third_party:nppidei",
        "@com_nvidia_isaac//third_party:nppif",
        "@com_nvidia_isaac//third_party:nppist",
        "@com_nvidia_isaac//third_party:nppisu",
        "@com_nvidia_isaac//third_party:npps",
        "@libjpeg//:jpeg",
        "@libusb",
        "@openmp//:openmp_dynamic",
    ],
)
