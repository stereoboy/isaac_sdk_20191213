"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

workspace(name = "com_nvidia_isaac")

load("//engine/build:isaac.bzl", "isaac_git_repository", "isaac_new_http_archive")
load("//third_party:engine.bzl", "isaac_engine_workspace")
load("//third_party:packages.bzl", "isaac_packages_workspace")
load("//third_party:ros.bzl", "isaac_ros_workspace")
load("//third_party:zed.bzl", "isaac_zed_workspace")

isaac_engine_workspace()

isaac_packages_workspace()

isaac_ros_workspace()

isaac_zed_workspace()

####################################################################################################
# Load cartographer

isaac_git_repository(
    name = "com_github_googlecartographer_cartographer",
    commit = "b6b41e9b173ea2e49e606f1e0d54d6d57ed421e3",
    licenses = ["@com_github_googlecartographer_cartographer//:LICENSE"],
    remote = "https://github.com/googlecartographer/cartographer.git",
)

isaac_git_repository(
    name = "com_google_protobuf",
    commit = "48cb18e5c419ddd23d9badcfe4e9df7bde1979b2",
    licenses = ["@com_google_protobuf//:LICENSE"],
    remote = "https://github.com/protocolbuffers/protobuf.git",
)

isaac_new_http_archive(
    name = "org_lzma_lzma",
    build_file = "@com_nvidia_isaac//third_party:lzma.BUILD",
    licenses = ["@org_lzma_lzma//:COPYING"],
    sha256 = "9717ae363760dedf573dad241420c5fea86256b65bc21d2cf71b2b12f0544f4b",
    strip_prefix = "xz-5.2.4",
    type = "tar.xz",
    url = "https://developer.nvidia.com/isaac/download/third_party/xz-5-2-4-tar-xz",
)

load("@com_github_googlecartographer_cartographer//:bazel/repositories.bzl", "cartographer_repositories")

cartographer_repositories()

# Loads boost c++ library (https://www.boost.org/) and
# custom bazel build support (https://github.com/nelhage/rules_boost/)
# explicitly for cartographer
# due to bazel bug: https://github.com/bazelbuild/bazel/issues/1550
isaac_git_repository(
    name = "com_github_nelhage_rules_boost",
    commit = "82ae1790cef07f3fd618592ad227fe2d66fe0b31",
    licenses = ["@com_github_nelhage_rules_boost//:LICENSE"],
    remote = "https://github.com/nelhage/rules_boost.git",
)

load("@com_github_nelhage_rules_boost//:boost/boost.bzl", "boost_deps")

boost_deps()

# Loads Google grpc C++ library (https://grpc.io/) explicitly for cartographer
load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# Loads Prometheus Data Model Client Library (https://github.com/jupp0r/prometheus-cpp/) and
# CivetWeb (https://github.com/civetweb/civetweb/) explicitly for cartographer
load("@com_github_jupp0r_prometheus_cpp//:repositories.bzl", "load_civetweb", "load_prometheus_client_model")

load_civetweb()

load_prometheus_client_model()

####################################################################################################

# Configures toolchain
load("//engine/build/toolchain:toolchain.bzl", "toolchain_configure")

toolchain_configure(name = "toolchain")
