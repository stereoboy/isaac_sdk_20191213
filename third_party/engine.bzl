"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_git_repository", "isaac_new_git_repository")
load("//engine/build:isaac.bzl", "isaac_http_archive", "isaac_new_http_archive")
load("//engine/build:isaac.bzl", "isaac_new_local_repository")

def clean_dep(dep):
    return str(Label(dep))

# loads dependencies required to build apps with alice
def isaac_engine_workspace():
    isaac_new_http_archive(
        name = "gtest",
        build_file = clean_dep("//third_party:gtest.BUILD"),
        sha256 = "d88ad7eba129d2d5453da05c6318af6babf65af37835d720e6bfa105d61cf5ce",
        url = "https://developer.nvidia.com/isaac/download/third_party/googletest-release-1-8-0-tar-gz",
        type = "tar.gz",
        strip_prefix = "googletest-release-1.8.0/googletest",
        licenses = ["@gtest//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "benchmark",
        build_file = clean_dep("//third_party:benchmark.BUILD"),
        sha256 = "f19559475a592cbd5ac48b61f6b9cedf87f0b6775d1443de54cfe8f53940b28d",
        url = "https://developer.nvidia.com/isaac/download/third_party/benchmark-1-3-0-tar-gz",
        type = "tar.gz",
        strip_prefix = "benchmark-1.3.0",
        licenses = ["@benchmark//:LICENSE"],
    )

    isaac_new_http_archive(
        # Name is matching that of deps for cartographer (@cartographer//bazel:repositories.bzl)
        name = "net_zlib_zlib",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        url = "https://developer.nvidia.com/isaac/download/third_party/zlib-1-2-11-tar-gz",
        type = "tar.gz",
        strip_prefix = "zlib-1.2.11",
        licenses = ["https://zlib.net/zlib_license.html"],
    )

    isaac_http_archive(
        name = "boringssl",
        sha256 = "524ba98a56300149696481b4cb9ddebd0c7b7ac9b9f6edee81da2d2d7e5d2bb3",
        url = "https://developer.nvidia.com/isaac/download/third_party/boringssl-a0fb951d2a26a8ee746b52f3ba81ab011a0af778-tar-gz",
        type = "tar.gz",
        patches = [clean_dep("//third_party:boringssl.patch")],
        strip_prefix = "boringssl-a0fb951d2a26a8ee746b52f3ba81ab011a0af778",
        licenses = ["@boringssl//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "libuuid",
        build_file = clean_dep("//third_party:uuid.BUILD"),
        sha256 = "46af3275291091009ad7f1b899de3d0cea0252737550e7919d17237997db5644",
        url = "https://developer.nvidia.com/isaac/download/third_party/libuuid-1-0-3-tar-gz",
        type = "tar.gz",
        strip_prefix = "libuuid-1.0.3",
        licenses = ["@libuuid//:COPYING"],
    )

    isaac_new_http_archive(
        # Name is matching that of deps for cartographer (@cartographer//bazel:repositories.bzl)
        name = "org_tuxfamily_eigen",
        build_file = clean_dep("//third_party:eigen.BUILD"),
        sha256 = "e5a4f7699e9379e5ce049000b937cf30dc1c22740667dfa5977b0ec13060d8ed",
        # This zip has been manually updated to fix some warnings:
        #  Eigen/Core was including `host_defines.h` directly instead of `cuda_runtime_api.h`
        url = "https://developer.nvidia.com/isaac/download/third_party/eigen-eigen-323c052e1731-v2-tar-bz2",
        type = "tar.bz2",
        strip_prefix = "eigen-eigen-323c052e1731",
        licenses = [
            "@org_tuxfamily_eigen/COPYING.BSD",
            "@org_tuxfamily_eigen/COPYING.MPL2",
            "@org_tuxfamily_eigen/COPYING.MINPACK",
            "@org_tuxfamily_eigen/COPYING.LGPL",
        ],
    )

    isaac_new_http_archive(
        # Name is matching that of deps for cartographer (@cartographer//bazel:repositories.bzl)
        name = "org_libpng_libpng",
        build_file = clean_dep("//third_party:png.BUILD"),
        sha256 = "2f1e960d92ce3b3abd03d06dfec9637dfbd22febf107a536b44f7a47c60659f6",
        url = "https://developer.nvidia.com/isaac/download/third_party/libpng-1-6-34-tar-xz",
        type = "tar.xz",
        strip_prefix = "libpng-1.6.34",
        licenses = ["@org_libpng_libpng//:libpng-LICENSE.txt"],
    )

    isaac_new_git_repository(
        name = "capnproto",
        build_file = clean_dep("//third_party:capnproto.BUILD"),
        tag = "v0.7.0",
        remote = "https://github.com/capnproto/capnproto.git",
        licenses = ["@capnproto//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "asio",
        build_file = clean_dep("//third_party:asio.BUILD"),
        sha256 = "4e27dcb37456ba707570334b91f4798721111ed67b69915685eac141895779aa",
        url = "https://developer.nvidia.com/isaac/download/third_party/asio-1-12-2-tar-bz2",
        type = "tar.bz2",
        strip_prefix = "asio-1.12.2",
        licenses = ["@asio//:asio-1.12.2/LICENSE_1_0.txt"],
    )

    isaac_git_repository(
        name = "com_github_gflags_gflags",
        commit = "e292e0452fcfd5a8ae055b59052fc041cbab4abf",
        remote = "https://github.com/gflags/gflags.git",
        licenses = ["@com_github_gflags_gflags//:COPYING.txt"],
    )

    native.bind(
        name = "gflags",
        actual = "@com_github_gflags_gflags//:gflags",
    )

    isaac_new_http_archive(
        name = "nlohmann_json",
        build_file = clean_dep("//third_party:nlohmann_json.BUILD"),
        sha256 = "e8fffa6cbdb3c15ecdff32eebf958b6c686bc188da8ad5c6489462d16f83ae54",
        url = "https://developer.nvidia.com/isaac/download/third_party/json-3-1-2-tar-gz",
        type = "tar.gz",
        licenses = ["@nlohmann_json//:LICENSE.MIT"],
    )

    isaac_new_http_archive(
        name = "lmdb",
        build_file = clean_dep("//third_party:lmdb.BUILD"),
        sha256 = "44602436c52c29d4f301f55f6fd8115f945469b868348e3cddaf91ab2473ea26",
        url = "https://github.com/LMDB/lmdb/archive/LMDB_0.9.24.tar.gz",
        type = "tar.gz",
        strip_prefix = "lmdb-LMDB_0.9.24",
        licenses = ["@lmdb//:libraries/liblmdb/LICENSE"],
    )

    isaac_new_http_archive(
        name = "nasm",
        build_file = clean_dep("//third_party:nasm.BUILD"),
        sha256 = "00b0891c678c065446ca59bcee64719d0096d54d6886e6e472aeee2e170ae324",
        url = "https://developer.nvidia.com/isaac/download/third_party/nasm-2-12-02-tar-bz2",
        type = "tar.bz2",
        strip_prefix = "nasm-2.12.02",
        licenses = ["@nasm//:LICENSE"],
    )

    isaac_new_http_archive(
        # Name is matching that of deps for cartographer (@cartographer//bazel:repositories.bzl)
        name = "libjpeg",
        build_file = clean_dep("//third_party:jpeg.BUILD"),
        sha256 = "c15a9607892113946379ccea3ca8b85018301b200754f209453ab21674268e77",
        url = "https://developer.nvidia.com/isaac/download/third_party/libjpeg-turbo-1-5-1-tar-gz",
        type = "tar.gz",
        strip_prefix = "libjpeg-turbo-1.5.1",
        licenses = ["@libjpeg//:LICENSE.md"],
    )

    isaac_new_http_archive(
        name = "uwebsockets",
        build_file = clean_dep("//third_party:uwebsockets.BUILD"),
        sha256 = "663a22b521c8258e215e34e01c7fcdbbd500296aab2c31d36857228424bb7675",
        url = "https://developer.nvidia.com/isaac/download/third_party/uwebsockets-0-14-8-tar-gz",
        type = "tar.gz",
        strip_prefix = "uWebSockets-0.14.8",
        licenses = ["@uwebsockets//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "snappy",
        build_file = clean_dep("//third_party:snappy.BUILD"),
        sha256 = "3dfa02e873ff51a11ee02b9ca391807f0c8ea0529a4924afa645fbf97163f9d4",
        url = "https://developer.nvidia.com/isaac/download/third_party/snappy-1-1-7-tar-gz",
        type = "tar.gz",
        licenses = ["@snappy//:COPYING"],
    )

    isaac_new_local_repository(
        name = "python",
        build_file = clean_dep("//third_party:python.BUILD"),
        path = "/usr",
        licenses = ["https://docs.python.org/3/license.html"],
    )

    isaac_new_http_archive(
        name = "pybind11",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
        sha256 = "0f34838f2c8024a6765168227ba587b3687729ebf03dc912f88ff75c7aa9cfe8",
        url = "https://developer.nvidia.com/isaac/download/third_party/pybind11-2-3-0-tar-gz",
        type = "tar.gz",
        strip_prefix = "pybind11-2.3.0",
        licenses = ["@pybind11//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "python_aarch64",
        build_file = clean_dep("//third_party:python_aarch64.BUILD"),
        sha256 = "0557f47820f90d0dc9371c991ecb361f0e5fbac753a416a80ed4ee7ad80906e6",
        url = "https://developer.nvidia.com/isaac/download/third_party/python3_aarch64_jp42-tar-xz",
        type = "tar.xz",
        licenses = ["https://docs.python.org/3/license.html"],
    )

    isaac_new_http_archive(
        name = "curl",
        build_file = clean_dep("//third_party:curl.BUILD"),
        sha256 = "ff3e80c1ca6a068428726cd7dd19037a47cc538ce58ef61c59587191039b2ca6",
        url = "https://developer.nvidia.com/isaac/download/third_party/curl-7-49-1-tar-gz",
        type = "tar.gz",
        strip_prefix = "curl-7.49.1",
        licenses = ["@curl//:COPYING"],
    )

    isaac_new_http_archive(
        name = "glfw",
        build_file = clean_dep("//third_party:glfw.BUILD"),
        sha256 = "e10f0de1384d75e6fc210c53e91843f6110d6c4f3afbfb588130713c2f9d8fe8",
        url = "https://developer.nvidia.com/isaac/download/third_party/glfw-3-2-1-tar-gz",
        type = "tar.gz",
        strip_prefix = "glfw-3.2.1",
        licenses = ["@glfw//:COPYING.txt"],
    )

    isaac_new_http_archive(
        name = "gl3w",
        build_file = clean_dep("//third_party:gl3w.BUILD"),
        sha256 = "442801ac9f10258499259b0b7679d28a1c1bf17e4a92c7e774b44bd4ba37525c",
        url = "https://developer.nvidia.com/isaac/download/third_party/gl3w-4f1d558410b0938840dc3db98e741d71f382ba22-tar-gz",
        type = "tar.gz",
        strip_prefix = "gl3w-4f1d558410b0938840dc3db98e741d71f382ba22",
        licenses = ["@gl3w//:UNLICENSE"],
    )

    isaac_new_http_archive(
        name = "imgui",
        build_file = clean_dep("//third_party:imgui.BUILD"),
        sha256 = "dc48173f9b34c763005e63dccb4355653909b25e04d5bc28ea9351540c457d96",
        url = "https://developer.nvidia.com/isaac/download/third_party/imgui-1-66-tar-gz",
        type = "tar.gz",
        strip_prefix = "imgui-1.66",
        licenses = ["@imgui//:LICENSE.txt"],
    )

    isaac_new_http_archive(
        name = "lss",
        build_file = clean_dep("//third_party:lss.BUILD"),
        sha256 = "6d2e98e9d360797db6348ae725be901c1947e5736d87f07917c2bd835b03eeef",
        url = "https://developer.nvidia.com/isaac/download/third_party/linux-syscall-support-93426bda6535943ff1525d0460aab5cc0870ccaf-tar-gz",
        type = "tar.gz",
        licenses = ["@lss//:linux_syscall_support.h"],
    )

    isaac_new_git_repository(
        name = "breakpad",
        build_file = clean_dep("//third_party:breakpad.BUILD"),
        remote = "https://chromium.googlesource.com/breakpad/breakpad",
        commit = "db1cda26539c711c3da7ed4d410dfe8190e89b8f",
        licenses = ["@breakpad//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "nvcc_10",
        build_file = clean_dep("//third_party:nvcc_10.BUILD"),
        sha256 = "91061a7a475b42557ce8ddad1337626624e449daed1e8bbce6a70d6b468df973",
        url = "https://developer.nvidia.com/isaac/download/third_party/cuda_10_nvcc-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    isaac_new_http_archive(
        name = "cuda_x86_64",
        build_file = clean_dep("//third_party:cuda_x86_64.BUILD"),
        sha256 = "f0ab78d7ca659dd70b1b9b6aefa6340fb03af474fed338a277942d0d7c70cef0",
        url = "https://developer.nvidia.com/isaac/download/third_party/cuda-10-0-cudnn7-6-3-28-x86_64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    isaac_new_http_archive(
        name = "cuda_aarch64_jetpack43",
        build_file = clean_dep("//third_party:cuda_aarch64_jetpack43.BUILD"),
        sha256 = "11919eeb2ab467bb32e87f7a33edd56ad18329aef4e97b94b1e1a8c57a62266d",
        url = "https://developer.nvidia.com/isaac/download/third_party/cuda-10-0_cudnn7-6-3-28-aarch64-tar-xz",
        type = "tar.xz",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )

    isaac_new_http_archive(
        name = "isaac_assets",
        url = "https://developer.nvidia.com/isaac/download/third_party/isaac_assets-zip",
        build_file = clean_dep("//third_party:isaac_assets.BUILD"),
        type = "zip",
        licenses = ["http://docs.nvidia.com/cuda/eula/index.html"],
    )
