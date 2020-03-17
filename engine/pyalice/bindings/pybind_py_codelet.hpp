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
#include "engine/pyalice/bindings/pybind_node.hpp"

namespace isaac {
namespace alice {

class PyCodelet;

// Provides access to alice pycodelet in Python
class PybindPyCodelet {
 public:
  PybindPyCodelet(alice::PybindNode node);

  // get isaac parameters which is serialized to a json string
  pybind11::str getConfig();
  // set isaac parameters by passing a json string
  void setConfig(pybind11::str config);

  void tickPeriodically(pybind11::float_ interval);
  void tickBlocking();
  // set tick on message beahviour by the channel's tag
  void tickOnMessage(pybind11::str tag);
  // set synchronization behaviour by the channel's tags
  void synchronize(pybind11::str tag1, pybind11::str tag2);

  pybind11::float_ getTickTime();
  pybind11::float_ getTickDt();
  pybind11::bool_ isFirstTick();
  pybind11::int_ getTickCount();

  // adds a new rx message hook by its tag
  void addRxHook(pybind11::str rx_hook);
  // adds a new tx message hook by its tag
  void addTxHook(pybind11::str tx_hook);
  // receives a message (in python bytes) given the channel's tag
  pybind11::bytes receive(pybind11::str tag);
  // publishes a message given the channel's tag and optional acqtime
  pybind11::dict publish(pybind11::bytes message, pybind11::str tag, pybind11::object acqtime);
  // checks if a message is available given the channel configuration dictionary
  pybind11::bool_ available(pybind11::str tag);
  // gets the pubtime of the message recieved in the channel
  pybind11::int_ getRxPubtime(pybind11::str tag);
  // get the acqtime of the message recieved in the channel
  pybind11::int_ getRxAcqtime(pybind11::str tag);
  // gets the uuid of the message recieved in the channel
  pybind11::str getRxUuid(pybind11::str tag);
  // gets the RX hook buffer content
  pybind11::bytes getRxBufferContent(pybind11::str tag, pybind11::int_ idx);
  // Adds numpy buffer for publishing with message. Returns total number of buffers that are
  // ready.
  pybind11::int_ addTxBuffer(pybind11::str tx_hook, pybind11::buffer b);

  // a blocking function that waits until the C++ side notifies it to start a particular job
  pybind11::str pythonWaitForJob();
  // a non-blocking function that notifies the C++ side that it finishes the job
  void pythonJobFinished();

  // publish on sight interface
  void show(pybind11::str sop_json);

 private:
  PyCodelet* pycodelet_node_;  // the actual C++ component that is registered as a isaac codelet
};

// Initializes the python module
void InitPybindPyCodelet(pybind11::module& m);

}  // namespace alice
}  // namespace isaac
