/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <pybind11/pybind11.h>

namespace isaac {
namespace alice {

class Node;
enum class Status;

// Provides access to alice nodes in Python
class PybindNode {
 public:
  PybindNode();
  PybindNode(alice::Node* node);
  ~PybindNode();

  // The node held by the wrapper
  alice::Node* handle() const { return node_; }
  // Checks if the alice node (node_) is valid (not nullptr)
  bool isValid() const { return node_ != nullptr; }
  // Get the status of the alice node (node_). A wrapper for the alice node's getStatus()
  alice::Status getStatus() const;

 private:
  alice::Node* node_;
};

// Initializes the python module
void InitPybindNode(pybind11::module& m);

}  // namespace alice
}  // namespace isaac
