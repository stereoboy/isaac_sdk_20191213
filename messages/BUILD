"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_library")
load("//messages:messages.bzl", "create_message_proto_libraries", "message_proto_library_names")

filegroup(
    name = "capnp",
    srcs = glob(["*.capnp"]),
    visibility = ["//visibility:public"],
)

# Update //messages:messages.bzl if you change the proto dependencies.
create_message_proto_libraries()

isaac_cc_library(
    name = "camera",
    hdrs = ["camera.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "camera_proto",
        "//engine/gems/geometry:pinhole",
    ],
)

isaac_cc_library(
    name = "element_type",
    hdrs = ["element_type.hpp"],
    srcs = ["element_type.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "element_type_proto",
        "//engine/core/tensor:element_type",
        "//engine/gems/math",
    ],
)

isaac_cc_library(
    name = "geometry",
    hdrs = ["geometry.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "geometry_proto",
        "//engine/gems/geometry:n_cuboid",
        "//engine/gems/geometry:plane",
    ],
)

isaac_cc_library(
    name = "image",
    hdrs = ["image.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "element_type",
        "image_proto",
        "//engine/gems/image",
    ],
)

isaac_cc_library(
    name = "json",
    hdrs = ["json.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "json_proto",
        "//engine/gems/serialization",
    ],
)

isaac_cc_library(
    name = "math",
    hdrs = ["math.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "math_proto",
        "//engine/gems/math",
    ],
)

isaac_cc_library(
    name = "sample_cloud",
    hdrs = ["sample_cloud.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "element_type",
        "sample_cloud_proto",
        "//engine/core",
        "//engine/core/buffers",
        "//engine/core/sample_cloud",
        "//engine/gems/math",
    ],
)

isaac_cc_library(
    name = "tensor",
    hdrs = ["tensor.hpp"],
    srcs = ["tensor.cpp"],
    visibility = ["//visibility:public"],
    deps = [
        "element_type",
        "tensor_proto",
        "//engine/core",
        "//engine/core/buffers",
        "//engine/core/tensor",
        "//engine/core/tensor:universal_tensor",
        "//engine/gems/math",
    ],
)

isaac_cc_library(
    name = "uuid",
    hdrs = ["uuid.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "uuid_proto",
        "//engine/gems/uuid",
    ],
)

isaac_cc_library(
    name = "messages",
    hdrs = ["messages.hpp"],
    visibility = ["//visibility:public"],
    deps = [
        "camera",
        "geometry",
        "image",
        "json",
        "math",
        "sample_cloud",
        "tensor",
        "uuid",
    ] + message_proto_library_names(),
)

isaac_cc_library(
    name = "proto_registry",
    srcs = [
        "proto_registry.cpp",
    ],
    hdrs = [
        "proto_registry.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        ":messages",
        "//engine/alice:message",
    ],
)
