"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_library")

isaac_cc_library(
    name = "object_detection_utils",
    srcs = ["object_detection_utils.cpp"],
    hdrs = [
        "bounding_box_detection.hpp",
        "object_detection_utils.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/gems/geometry:n_cuboid",
    ],
)

isaac_cc_library(
    name = "segmentation_utils",
    srcs = ["segmentation_utils.cpp"],
    hdrs = [
        "segmentation_utils.hpp",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//engine/core",
        "//engine/core/tensor",
        "//engine/gems/math",
    ],
)

isaac_cc_library(
    name = "cosine_similarity",
    srcs = ["cosine_similarity.cpp"],
    hdrs = ["cosine_similarity.hpp"],
    deps = [
        "//engine/core",
        "//engine/gems/tensor",
        "//third_party:cublas",
        "//third_party:cuda",
        "//third_party:cudart",
    ],
    visibility = ["//visibility:public"],
)