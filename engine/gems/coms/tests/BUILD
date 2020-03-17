"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

cc_test(
    name = "serial",
    srcs = ["serial.cpp"],
    tags = ["manual"],
    deps = [
        "//engine/core",
        "//engine/gems/coms:serial",
        "//engine/gems/math",
        "@gtest//:main",
    ],
)

cc_test(
    name = "i2c",
    srcs = ["i2c.cpp"],
    tags = ["manual"],
    deps = [
        "//engine/gems/coms:i2c",
        "@gtest//:main",
    ],
)
