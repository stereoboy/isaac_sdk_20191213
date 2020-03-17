/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <memory>

#include <pybind11/pybind11.h>
#include "engine/pyalice/bindings/pybind_node.hpp"

namespace isaac {
namespace alice {

class Application;

// Owns and provides access to an alice application in Python
// Unless specified otherwise function comments are the same as in engine/alice/application.hpp
class PybindApplication {
 public:
  PybindApplication(const std::string& app_filename, const std::string& more_jsons);
  ~PybindApplication();

  // The alice application held by the wrapper
  Application& app() const { return *app_; }

  void start();
  void start_wait_stop();
  void stop();
  pybind11::str uuid() const;
  PybindNode findNodeByName(const std::string& name);

 private:
  std::unique_ptr<Application> app_;
};

// Initializes the python module
void InitPybindApplication(pybind11::module& m);

}  // namespace alice
}  // namespace isaac
