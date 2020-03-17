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

major_version: "1"
minor_version: "0"
default_target_cpu: "same_as_host"

default_toolchain {
  cpu: "k8"
  toolchain_identifier: "local"
}

default_toolchain {
  cpu: "arm64-v8a"
  toolchain_identifier: "stub_arm64-v8a"
}

# Android tooling requires a default toolchain for the arm64-v8a cpu.
toolchain {
  abi_version: "arm64-v8a"
  abi_libc_version: "arm64-v8a"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "arm64-v8a"
  needsPic: true
  supports_gold_linker: false
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: false
  target_libc: "arm64-v8a"
  target_cpu: "arm64-v8a"
  target_system_name: "arm64-v8a"
  toolchain_identifier: "stub_arm64-v8a"

  cxx_flag: "-std=c++14"
  linker_flag: "-lstdc++"
  linker_flag: "-Wl,--dynamic-linker=/lib/ld-linux-aarch64.so.1"
  linker_flag: "-lm"
  linker_flag: "-fPIC"

  compiler_flag: "-D_DEFAULT_SOURCE"
  compiler_flag: "-U_FORTIFY_SOURCE"
  # Specific for glibc https://en.cppreference.com/w/cpp/types/integer
  compiler_flag: "-D__STDC_FORMAT_MACROS"
  compiler_flag: "-fPIC"

  cxx_builtin_include_directory: "/usr/aarch64-linux-gnu/include"
  cxx_builtin_include_directory: "/usr/aarch64-linux-gnu/include/c++/7"
  cxx_builtin_include_directory: "/usr/lib/gcc-cross/aarch64-linux-gnu/7/include"
  cxx_builtin_include_directory: "/usr/lib/gcc-cross/aarch64-linux-gnu/7/include-fixed"
  cxx_builtin_include_directory: "/usr/aarch64-linux-gnu/include/c++/7/backward"
  cxx_builtin_include_directory: "/usr/aarch64-linux-gnu/include"

  tool_path {name: "ld" path: "/usr/bin/aarch64-linux-gnu-ld" }
  tool_path {name: "cpp" path: "/usr/bin/aarch64-linux-gnu-cpp-5" }
  tool_path {name: "dwp" path: "/usr/bin/aarch64-linux-gnu-dwp" }
  tool_path {name: "gcov" path: "/usr/bin/aarch64-linux-gnu-gcov" }
  tool_path {name: "nm" path: "/usr/bin/aarch64-linux-gnu-nm" }
  tool_path {name: "objcopy" path: "/usr/bin/aarch64-linux-gnu-objcopy" }
  tool_path {name: "objdump" path: "/usr/bin/aarch64-linux-gnu-objdump" }
  tool_path {name: "strip" path: "/usr/bin/aarch64-linux-gnu-strip" }
  tool_path {name: "gcc" path: "scripts/crosstool_wrapper_driver_is_not_gcc.py" }
  tool_path {name: "ar" path: "/usr/bin/aarch64-linux-gnu-ar" }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-Og"
    compiler_flag: "-ggdb3"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }
  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O2"
    compiler_flag: "-ggdb2"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }
  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-O2"
    compiler_flag: "-ggdb3"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }

  linking_mode_flags { mode: DYNAMIC }
}

toolchain {
  toolchain_identifier: "local"
  abi_version: "local"
  abi_libc_version: "local"
  builtin_sysroot: ""
  compiler: "compiler"
  host_system_name: "local"
  needsPic: true
  supports_gold_linker: true
  supports_incremental_linker: false
  supports_fission: false
  supports_interface_shared_objects: false
  supports_normalizing_ar: false
  supports_start_end_lib: true
  target_libc: "local"
  target_cpu: "k8"
  target_system_name: "local"
  cxx_flag: "-std=c++14"
  linker_flag: "-lstdc++"
  linker_flag: "-lm"
  linker_flag: "-fuse-ld=gold"
  linker_flag: "-Wl,-no-as-needed"
  linker_flag: "-Wl,-z,relro,-z,now"
  linker_flag: "-B/usr/bin"
  linker_flag: "-pass-exit-codes"
  linker_flag: "-fPIC"
  cxx_builtin_include_directory: "/usr/include/c++/7"
  cxx_builtin_include_directory: "/usr/include/x86_64-linux-gnu/c++/7"
  cxx_builtin_include_directory: "/usr/include/c++/7/backward"
  cxx_builtin_include_directory: "/usr/lib/gcc/x86_64-linux-gnu/7/include"
  cxx_builtin_include_directory: "/usr/local/include"
  cxx_builtin_include_directory: "/usr/lib/gcc/x86_64-linux-gnu/7/include-fixed"
  cxx_builtin_include_directory: "/usr/include/x86_64-linux-gnu"
  cxx_builtin_include_directory: "/usr/include"
  objcopy_embed_flag: "-I"
  objcopy_embed_flag: "binary"
  unfiltered_cxx_flag: "-fno-canonical-system-headers"
  unfiltered_cxx_flag: "-Wno-builtin-macro-redefined"
  unfiltered_cxx_flag: "-D__DATE__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIMESTAMP__=\"redacted\""
  unfiltered_cxx_flag: "-D__TIME__=\"redacted\""
  compiler_flag: "-D_DEFAULT_SOURCE"
  compiler_flag: "-U_FORTIFY_SOURCE"
  compiler_flag: "-fstack-protector"
  compiler_flag: "-Wall"
  compiler_flag: "-Werror"
  compiler_flag: "-B/usr/bin"
  compiler_flag: "-Wunused-but-set-parameter"
  compiler_flag: "-Wno-free-nonheap-object"
  compiler_flag: "-fno-omit-frame-pointer"
  compiler_flag: "-fPIC"
  # Specific for glibc https://en.cppreference.com/w/cpp/types/integer
  compiler_flag: "-D__STDC_FORMAT_MACROS"
  tool_path {name: "ld" path: "/usr/bin/ld" }
  tool_path {name: "cpp" path: "/usr/bin/cpp" }
  tool_path {name: "dwp" path: "/usr/bin/dwp" }
  tool_path {name: "gcov" path: "/usr/bin/gcov" }
  tool_path {name: "nm" path: "/usr/bin/nm" }
  tool_path {name: "objcopy" path: "/usr/bin/objcopy" }
  tool_path {name: "objdump" path: "/usr/bin/objdump" }
  tool_path {name: "strip" path: "/usr/bin/strip" }
  tool_path {name: "gcc" path: "scripts/crosstool_wrapper_driver_is_not_gcc_host.py" }
  tool_path {name: "ar" path: "/usr/bin/ar" }

  compilation_mode_flags {
    mode: DBG
    compiler_flag: "-Og"
    compiler_flag: "-ggdb3"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }
  compilation_mode_flags {
    mode: FASTBUILD
    compiler_flag: "-O2"
    compiler_flag: "-ggdb2"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }
  compilation_mode_flags {
    mode: OPT
    compiler_flag: "-O2"
    compiler_flag: "-ggdb2"
    compiler_flag: "-D_FORTIFY_SOURCE=2"
    compiler_flag: "-DNDEBUG"
    compiler_flag: "-ffunction-sections"
    compiler_flag: "-fdata-sections"
    linker_flag: "-Wl,--gc-sections"
  }
  linking_mode_flags { mode: DYNAMIC }

  feature {
    name: 'coverage'
    provides: 'profile'
    flag_set {
      action: 'preprocess-assemble'
      action: 'c-compile'
      action: 'c++-compile'
      action: 'c++-header-parsing'
      action: 'c++-header-preprocessing'
      action: 'c++-module-compile'
      flag_group {
        flag: '-fprofile-arcs'
        flag: '-ftest-coverage'
      }
    }
    flag_set {
      action: 'c++-link-interface-dynamic-library'
      action: 'c++-link-dynamic-library'
      action: 'c++-link-executable'
      flag_group {
        flag: '-lgcov'
      }
    }
  }
}
