"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_module", "isaac_component")

isaac_cc_module(
    name = "components",
    deps = [
        ":dummy_pose2d_codelets",
        ":dummy_status",
    ],
)

isaac_component(
    name = "dummy_pose2d_codelets",
    filename = "dummy_pose2d_codelets",
)

isaac_component(
    name = "dummy_status",
    filename = "dummy_status",
)

py_test(
    name = "pycodelet_test",
    srcs = ["pycodelet_test.py"],
    data = [
        "pycodelet_test_cpp2py.app.json",
        "pycodelet_test_img.app.json",
        "pycodelet_test_py2cpp.app.json",
        "pycodelet_test_py2py.app.json",
        "pycodelet_test_sync.app.json",
        "//engine/pyalice/tests:libcomponents_module.so",
        "//messages:camera_proto",
        "//messages:math_proto",
        "//messages:uuid_proto",
        "//packages/message_generators:libmessage_generators_module.so",
        "@isaac_assets",
    ],
    deps = [
        "//engine/pyalice",
    ],
)

py_test(
    name = "pynode_test",
    srcs = ["pynode_test.py"],
    data = [
        "pynode_failure.json",
        "pynode_success.json",
        "pynode_test_status.app.json",
        "//engine/pyalice/tests:libcomponents_module.so",
        "//packages/message_generators:libmessage_generators_module.so",
    ],
    deps = [
        "//engine/pyalice",
    ],
)

py_test(
    name = "buffer_test",
    srcs = ["buffer_test.py"],
    data = [
        "buffer_test.app.json",
        "//engine/pyalice/tests:libcomponents_module.so",
        "//messages:camera_proto",
        "//packages/message_generators:libmessage_generators_module.so",
        "@ml_test_data",
    ],
    deps = [
        "//engine/pyalice",
    ],
)
