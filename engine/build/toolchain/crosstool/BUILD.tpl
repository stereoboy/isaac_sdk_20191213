# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "empty",
    srcs = [],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# the "${CPU}" entry in the toolchains attribute.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "k8|compiler": ":cc-compiler-k8",
        "arm64-v8a|compiler": ":cc-compiler-arm64-v8a",
    },
)

cc_toolchain(
    name = "cc-compiler-k8",
    all_files = ":gcc_or_nvcc",
    compiler_files = ":empty",
    cpu = "k8",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":gcc_or_nvcc",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
)

# Android tooling requires a default toolchain for the arm64-v8a cpu.
cc_toolchain(
    name = "cc-compiler-arm64-v8a",
    all_files = ":gcc_or_nvcc",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":gcc_or_nvcc",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
)

filegroup(
    name = "gcc_or_nvcc",
    srcs = [
        "scripts/crosstool_wrapper_driver_is_not_gcc.py",
        "scripts/crosstool_wrapper_driver_is_not_gcc_host.py",
        "@com_nvidia_isaac//engine/build:nvcc",
    ],
)
