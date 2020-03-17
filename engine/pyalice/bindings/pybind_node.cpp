/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "pybind_node.hpp"

#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

PybindNode::PybindNode() : node_(nullptr) {}

PybindNode::PybindNode(alice::Node* node) : node_(node) {}

PybindNode::~PybindNode() {}

alice::Status PybindNode::getStatus() const {
  return (node_ ? node_->getStatus() : Status::INVALID);
}

void InitPybindNode(pybind11::module& m) {
  pybind11::enum_<Status>(m, "Status")
      .value("Success", Status::SUCCESS)
      .value("Failure", Status::FAILURE)
      .value("Running", Status::RUNNING)
      .value("Invalid", Status::INVALID)
      .export_values();
  pybind11::class_<PybindNode>(m, "PybindNode")
      .def(pybind11::init<>())
      .def("is_valid", &isaac::alice::PybindNode::isValid)
      .def("get_status", &isaac::alice::PybindNode::getStatus);
}

}  // namespace alice
}  // namespace isaac
