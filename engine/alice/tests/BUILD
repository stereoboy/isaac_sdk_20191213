"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_cc_binary", "isaac_cc_library", "isaac_cc_test_group")
load("//engine/build:cc_capnp_library.bzl", "cc_capnp_library")

cc_capnp_library(
    name = "message_protos",
    protos = ["messages.capnp"],
)

cc_capnp_library(
    name = "foo_bar_protos",
    protos = [
        "bar.capnp",
        "foo.capnp",
    ],
)

isaac_cc_library(
    name = "messages",
    srcs = [
        "bulky_transmission.cpp",
        "foo_transmission.cpp",
    ],
    hdrs = [
        "bulky_transmission.hpp",
        "foo_transmission.hpp",
        "messages.hpp",
    ],
    deps = [
        ":message_protos",
        "//engine/alice",
        "@gtest//:main",
    ],
)

isaac_cc_test_group(
    srcs = [
        "app_load.cpp",
        "capnproto.cpp",
        "clock.cpp",
        "codelet_lifecycle.cpp",
        "codelet_registry.cpp",
        "codelet_start_if_no_trigger.cpp",
        "codelet_tick_period.cpp",
        "config.cpp",
        "config_immutability.cpp",
        "config_parallel.cpp",
        "connect_in_start.cpp",
        "create.cpp",
        "destroy_node.cpp",
        "dynamic_ping.cpp",
        "event_overload.cpp",
        "jammer.cpp",
        "live_node_creation.cpp",
        "message_hop.cpp",
        "message_loop.cpp",
        "network_large_blobs.cpp",
        "network_ping.cpp",
        "path_utils.cpp",
        "pose.cpp",
        "pose_to_message.cpp",
        "pose_tree.cpp",
        "proto_ping.cpp",
        "raw_ping.cpp",
        "record_replay_loop.cpp",
        "save_load_app.cpp",
        "stacktrace.cpp",
        "stresstest_ping.cpp",
        "synchronize.cpp",
        "the_fall.cpp",
        "throttle.cpp",
        "tick_tock.cpp",
        "time_offset.cpp",
        "variables.cpp",
    ],
    deps = [
        ":foo_bar_protos",
        ":messages",
        "//engine/alice",
        "//engine/alice/utils:path_utils",
    ],
)

# These tests are known to be flaky under system load. They are thus disabled from automatic
# commit validation for now.
isaac_cc_test_group(
    srcs = [
        "massively_ticking.cpp",
        "network_ping_pong.cpp",
        "ping_pong.cpp",
        "record_replay.cpp",
        "record_replay_large.cpp",
        "record_replay_multiple.cpp",
        "tcp_reconnect.cpp",
        "time_machine.cpp",
        "time_sync_apps.cpp",
    ],
    tags = ["manual"],
    deps = [
        ":foo_bar_protos",
        ":messages",
        "//engine/alice",
        "//engine/alice/utils:path_utils",
    ],
)

cc_test(
    name = "subgraphs",
    srcs = ["subgraphs.cpp"],
    data = glob([
        "subgraphs*.app.json",
        "transmission*.group.json",
    ]),
    deps = [
        ":messages",
        "//engine/alice",
        "@gtest//:main",
    ],
)

cc_test(
    name = "modules",
    srcs = ["modules.cpp"],
    data = [
        "//packages/message_generators:libmessage_generators_module.so",
        "//packages/flatsim:libflatsim_module.so",
    ],
    deps = [
        "//engine/alice",
        "@gtest//:main",
    ],
)

isaac_cc_library(
    name = "image_transmission",
    srcs = [
        "image_transmission.cpp",
    ],
    hdrs = [
        "image_transmission.hpp",
    ],
    deps = [
        "//engine/alice",
        "//messages:image",
        "@gtest//:main",
    ],
)

isaac_cc_test_group(
    srcs = [
        "message_images.cpp",
        "network_images_tcp.cpp",
        # "network_images_udp.cpp",  // UDP-based message passing not yet supported for images
        "record_replay_images.cpp",
    ],
    deps = [
        ":image_transmission",
        "//engine/alice",
    ],
)

isaac_cc_binary(
    name = "crashdump_binary",
    srcs = [
        "crashdump.cpp",
    ],
    deps = [
        "//engine/alice",
        "//engine/alice/tools:parse_command_line",
        "//external:gflags",
        "//messages",
    ],
)

sh_test(
    name = "crashdump_sh",
    size = "small",
    srcs = ["crashdump.sh"],
    data = [
        ":crashdump_binary",
        "@breakpad//:dump_syms",
        "@breakpad//:minidump_stackwalk",
    ],
    deps = [
    ],
)

cc_test(
    name = "proto_json",
    srcs = ["proto_json.cpp"],
    data = [
        "proto_json.app.json",
        "//packages/message_generators:libmessage_generators_module.so",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/alice",
        "//messages",
        "//messages:proto_registry",
        "@gtest//:main",
    ],
)
