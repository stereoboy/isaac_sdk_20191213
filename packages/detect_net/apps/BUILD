"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_app", "isaac_subgraph")

isaac_subgraph(
    name = "detect_net_inference",
    modules = [
        "detect_net",
        "ml",
        "perception",
        "viewers",
    ],
    subgraph = "detect_net_inference.subgraph.json",
    visibility = ["//visibility:public"],
)

isaac_app(
    name = "detect_net_inference_imagefeeder",
    data = [
        ":detect_net_inference",
        "@dolly_detection_pose_estimation_data",
        "@resnet_object_detection_model",
    ],
    modules = [
        "message_generators",
        "detect_net",
    ],
)

isaac_app(
    name = "detect_net_inference_camerafeed",
    data = [
        ":detect_net_inference",
        "@resnet_object_detection_model",
    ],
    modules = [
        "detect_net",
        "realsense",
    ],
)

isaac_app(
    name = "detect_net_inference_replay",
    data = [
        ":detect_net_inference",
        "//packages/record_replay/apps:replay_subgraph",
        "@dolly_detection_pose_estimation_data",
        "@resnet_object_detection_model",
    ],
    modules = [
        "detect_net",
    ],
)
