"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

CUDA_SO = [
    "cublas",
    "cudart",
    "cufft",
    "curand",
    "cusolver",
    "cusparse",
    "nvgraph",
    "nvrtc",
]

NPP_SO = [
    "nppc",
    "nppial",
    "nppicc",
    "nppicom",
    "nppidei",
    "nppif",
    "nppig",
    "nppim",
    "nppist",
    "nppisu",
    "nppitc",
    "npps",
]

# Get the path for the shared library with given name for the given version
def cuda_so_path(name, version):
    return "usr/local/cuda-" + version + "/lib64/lib" + name + ".so." + version

# Get the path for libcuda.so for the given version. A stub is used as the library is provided
# by the CUDA driver and is required to be available on the system.
def cuda_driver_so_path(version):
    return "usr/local/cuda-" + version + "/lib64/stubs/libcuda.so"

# Get the path for libnvToolsExt.so for the given version. A stub is used as the library is provided
# by the CUDA driver and is required to be available on the system.
def cuda_nv_tools_ext_so_path(version):
    return "usr/local/cuda-" + version + "/lib64/libnvToolsExt.so.1"

# Creates CUDA related dependencies. The arguments `family` and `version` are used to find the
# library and header files in the package
def cuda_device_deps(family, version):
    cuda_include_prefix = "usr/local/cuda-" + version + "/targets/" + family + "-linux/include"

    # CUDA
    cuda_hdrs = native.glob([
        # FIXME separate out headers
        cuda_include_prefix + "/*.h",
        cuda_include_prefix + "/*.hpp",
        cuda_include_prefix + "/CL/*.h",
        cuda_include_prefix + "/crt/*",
    ])

    # Create a stub library for the CUDA base library provided by the driver
    native.cc_library(
        name = "cuda",
        hdrs = cuda_hdrs,
        srcs = [cuda_driver_so_path(version), cuda_nv_tools_ext_so_path(version)],
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    # Create one library per CUDA shared libray
    for so in CUDA_SO:
        native.cc_library(
            name = so,
            hdrs = cuda_hdrs,
            srcs = [cuda_so_path(so, version)],
            strip_include_prefix = cuda_include_prefix,
            visibility = ["//visibility:public"],
        )

    # NPP
    npp_hdrs = native.glob([cuda_include_prefix + "/npp*.*"])  # FIXME separate out headers
    for so in NPP_SO:
        native.cc_library(
            name = so,
            hdrs = npp_hdrs,
            srcs = [cuda_so_path(so, version)],
            # Dependency graph: nppc <- npps <- everything else
            deps = ["cudart"] +
                   ["nppc"] if so != "nppc" else [] +
                                                 ["npps"] if so != "npps" and so != "nppc" else [],
            strip_include_prefix = cuda_include_prefix,
            visibility = ["//visibility:public"],
        )

    # THRUST
    native.cc_library(
        name = "thrust",
        hdrs = native.glob([cuda_include_prefix + "/thrust/**/*"]),
        deps = ["cudart"],
        strip_include_prefix = cuda_include_prefix,
        visibility = ["//visibility:public"],
    )

    # CUDNN
    cudnn_include_prefix = "usr/include/" + family + "-linux-gnu"
    native.cc_library(
        name = "cudnn",
        hdrs = [cudnn_include_prefix + "/cudnn.h"],
        includes = [cudnn_include_prefix],
        strip_include_prefix = cudnn_include_prefix,
        srcs = ["usr/lib/" + family + "-linux-gnu/libcudnn.so.7"],
        deps = ["cudart"],
        linkstatic = True,
        visibility = ["//visibility:public"],
    )

# Selects the correct version of `target` based on the current platform
def _cuda_select(target):
    return select({
        "//engine/build:platform_x86_64": ["@cuda_x86_64//:" + target],
        "//engine/build:platform_jetpack43": ["@cuda_aarch64_jetpack43//:" + target],
    })

# Creates all CUDA related dependencies for the current platform
def cuda_deps():
    TARGETS = ["cuda"] + CUDA_SO + NPP_SO + ["cudnn", "thrust"]
    for target in TARGETS:
        native.cc_library(
            name = target,
            visibility = ["//visibility:public"],
            deps = _cuda_select(target),
        )
