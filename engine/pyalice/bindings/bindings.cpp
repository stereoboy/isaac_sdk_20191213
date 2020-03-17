/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/pyalice/bindings/pybind_application.hpp"
#include "engine/pyalice/bindings/pybind_node.hpp"
#include "engine/pyalice/bindings/pybind_py_codelet.hpp"
#include <pybind11/pybind11.h>

PYBIND11_MODULE(bindings, m) {
  m.doc() = R"pbdoc(
        Isaac Alice Python Bridge
        -----------------------

        .. currentmodule:: pyalice

    )pbdoc";

  isaac::alice::InitPybindApplication(m);
  isaac::alice::InitPybindNode(m);
  isaac::alice::InitPybindPyCodelet(m);
}
