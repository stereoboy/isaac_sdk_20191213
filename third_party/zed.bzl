"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_new_http_archive")
load("//engine/build:isaac.bzl", "isaac_new_local_repository")

def clean_dep(dep):
    return str(Label(dep))

# loads dependencies for zed
def isaac_zed_workspace():
    isaac_new_http_archive(
        name = "libusb",
        build_file = clean_dep("//third_party:libusb.BUILD"),
        sha256 = "3500f7b182750cd9ccf9be8b1df998f83df56a39ab264976bdb3307773e16f48",
        url = "https://developer.nvidia.com/isaac/download/third_party/libusb-1-0-22-tar-gz",
        type = "tar.gz",
        strip_prefix = "libusb-1.0.22",
        licenses = ["@libusb//:COPYING"],
    )

    isaac_new_local_repository(
        name = "openmp",
        build_file = clean_dep("//third_party:openmp.BUILD"),
        path = "/usr/lib/",
        licenses = ["https://raw.githubusercontent.com/llvm/llvm-project/master/openmp/LICENSE.txt"],
    )

    isaac_new_local_repository(
        name = "libudev",
        build_file = clean_dep("//third_party:libudev.BUILD"),
        path = "/lib",
        licenses = ["https://raw.githubusercontent.com/systemd/systemd/master/LICENSE.LGPL2.1"],
    )

    isaac_new_http_archive(
        name = "libudev_aarch64",
        build_file = clean_dep("//third_party:libudev_aarch64.BUILD"),
        sha256 = "19634360f2b305d4d4ea883650c8bb6f1622d0f129d807274354fe7fc4d4eb33",
        url = "https://developer.nvidia.com/isaac/download/third_party/libudev-aarch64-tar-xz",
        type = "tar.xz",
        licenses = ["https://raw.githubusercontent.com/systemd/systemd/master/LICENSE.LGPL2.1"],
    )

    isaac_new_http_archive(
        name = "zed_x86_64",
        build_file = clean_dep("//third_party:zed_x86_64.BUILD"),
        sha256 = "dc7c0117f9e21fe3da2b65fbd724153aa2fddd054ae2f9935aa60ba3049fce30",
        url = "https://developer.nvidia.com/isaac/download/third_party/zed-2-8-3-x86_64-cuda10-tar-xz",
        type = "tar.xz",
        licenses = ["@zed_x86_64//:LICENSE.txt"],
    )

    isaac_new_http_archive(
        name = "zed_aarch64_jetpack42",
        build_file = clean_dep("//third_party:zed_jetpack42.BUILD"),
        sha256 = "91fc5c9b6857602f3bf5d5de8ae9d898dc9f53d475509cd48559473e9c5e45a5",
        url = "https://developer.nvidia.com/isaac/download/third_party/zed-2-8-3-aarch64-jp42-tar-xz",
        type = "tar.xz",
        licenses = ["@zed_aarch64_jetpack42//:LICENSE.txt"],
    )

    isaac_new_http_archive(
        name = "libpng12_x86_64",
        build_file = clean_dep("//third_party:libpng12_x86_64.BUILD"),
        sha256 = "e7f32a36bba77179a76df028221d7ba3408e6c34633f4e60e6936cb6231a665a",
        url = "https://developer.nvidia.com/isaac/download/third_party/libpng12_0_x86_64_xenial-tar-xz",
        type = "tar.xz",
        licenses = ["@libpng12_x86_64//:usr/share/doc/libpng12-0/copyright"],
    )
