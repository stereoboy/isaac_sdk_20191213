/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "pybind_py_codelet.hpp"

#include <pybind11/pybind11.h>

#include "engine/alice/components/PyCodelet.hpp"
#include "engine/alice/message.hpp"
#include "engine/core/assert.hpp"
#include "engine/gems/serialization/json.hpp"

namespace isaac {
namespace alice {

PybindPyCodelet::PybindPyCodelet(alice::PybindNode node) {
  ASSERT(node.handle(), "null argument");
  pycodelet_node_ = node.handle()->getComponent<PyCodelet>();
  ASSERT(pycodelet_node_, "No PyCodelet in node '%s'", node.handle()->name().c_str());
}

pybind11::str PybindPyCodelet::getConfig() {
  std::string result;
  {
    pybind11::gil_scoped_release release_gil;
    result = pycodelet_node_->get_config().dump();
  }
  return pybind11::str(result);
}

void PybindPyCodelet::setConfig(pybind11::str config) {
  const std::string config_string = config;
  {
    pybind11::gil_scoped_release release_gil;
    auto json_config = serialization::ParseJson(config_string);
    if (json_config) {
      pycodelet_node_->async_set_config(*json_config);
    } else {
      LOG_ERROR("invalid json format given to the PyCodelet");
    }
  }
}

void PybindPyCodelet::tickPeriodically(pybind11::float_ interval) {
  const double interval_val = interval;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->tickPeriodically(interval_val);
  }
}

void PybindPyCodelet::tickBlocking() {
  pycodelet_node_->tickBlocking();
}

void PybindPyCodelet::tickOnMessage(pybind11::str tag) {
  const std::string tag_val = std::string(tag);
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->tickOnMessageWithTag(tag);
  }
}

void PybindPyCodelet::synchronize(pybind11::str py_tag1, pybind11::str py_tag2) {
  const std::string tag1 = py_tag1;
  const std::string tag2 = py_tag2;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->synchronizeWithTags(tag1, tag2);
  }
}

pybind11::float_ PybindPyCodelet::getTickTime() {
  double val = 0.0;
  {
    pybind11::gil_scoped_release release_gil;
    val = pycodelet_node_->getTickTime();
  }
  return val;
}

pybind11::float_ PybindPyCodelet::getTickDt() {
  double val = 0.0;
  {
    pybind11::gil_scoped_release release_gil;
    val = pycodelet_node_->getTickDt();
  }
  return val;
}

pybind11::bool_ PybindPyCodelet::isFirstTick() {
  bool result = false;
  {
    pybind11::gil_scoped_release release_gil;
    result = pycodelet_node_->isFirstTick();
  }
  return result;
}

pybind11::int_ PybindPyCodelet::getTickCount() {
  int val = 0;
  {
    pybind11::gil_scoped_release release_gil;
    val = pycodelet_node_->getTickCount();
  }
  return val;
}

void PybindPyCodelet::addRxHook(pybind11::str rx_hook) {
  std::string hook = rx_hook;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->addRxHook(hook);
  }
}

void PybindPyCodelet::addTxHook(pybind11::str tx_hook) {
  std::string hook = tx_hook;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->addTxHook(hook);
  }
}

pybind11::bytes PybindPyCodelet::receive(pybind11::str tag) {
  std::string msg;
  pycodelet_node_->receive(pybind11::str(tag), msg);
  return pybind11::bytes(msg);
}

pybind11::dict PybindPyCodelet::publish(pybind11::bytes message, pybind11::str tag,
                                        pybind11::object py_acqtime) {
  const BufferedProtoMessage* proto_msg;
  std::optional<int64_t> acqtime(std::nullopt);
  if (!py_acqtime.is_none()) {
    acqtime = int64_t(pybind11::int_(py_acqtime));
  }

  {
    pybind11::gil_scoped_release release_gil;
    proto_msg = pycodelet_node_->publish(tag, message, acqtime);
  }

  pybind11::dict result;
  result["uuid"] = proto_msg->uuid.c_str();
  result["acqtime"] = proto_msg->acqtime;
  result["pubtime"] = proto_msg->pubtime;
  return result;
}

void PybindPyCodelet::show(pybind11::str sop_json) {
  std::string sop_string = sop_json;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->show(sop_string);
  }
}

pybind11::bool_ PybindPyCodelet::available(pybind11::str tag) {
  bool result = false;
  {
    pybind11::gil_scoped_release release_gil;
    result = pycodelet_node_->available(tag);
  }
  return result;
}

pybind11::int_ PybindPyCodelet::getRxPubtime(pybind11::str tag) {
  std::string tag_name = tag;
  int val = 0;
  {
    pybind11::gil_scoped_release release_gil;
    val = pycodelet_node_->getRxPubtime(tag);
  }
  return val;
}

pybind11::int_ PybindPyCodelet::getRxAcqtime(pybind11::str tag) {
  std::string tag_name = tag;
  int val = 0;
  {
    pybind11::gil_scoped_release release_gil;
    val = pycodelet_node_->getRxAcqtime(tag);
  }
  return val;
}

pybind11::str PybindPyCodelet::getRxUuid(pybind11::str tag) {
  std::string uuid;
  std::string tag_name = tag;
  {
    pybind11::gil_scoped_release release_gil;
    uuid = pycodelet_node_->getRxUuid(tag_name).c_str();
  }
  return pybind11::str(uuid);
}

pybind11::str PybindPyCodelet::pythonWaitForJob() {
  std::string job;
  {
    pybind11::gil_scoped_release release_gil;
    job = pycodelet_node_->pythonWaitForJob();
  }
  return pybind11::str(job);
}

void PybindPyCodelet::pythonJobFinished() {
  pycodelet_node_->pythonJobFinished();
}

pybind11::bytes PybindPyCodelet::getRxBufferContent(pybind11::str tag, pybind11::int_ idx) {
  std::string result;
  const std::string tag_name = tag;
  const int index = idx;
  {
    pybind11::gil_scoped_release release_gil;
    pycodelet_node_->getRxBufferWithTagAndIndex(tag_name, index, result);
  }
  return pybind11::bytes(result);
}

pybind11::int_ PybindPyCodelet::addTxBuffer(pybind11::str tx_hook, pybind11::buffer b) {
  const std::string hook_tag = tx_hook;
  ASSERT(!hook_tag.empty(), "Invalid tx tag");
  pybind11::buffer_info info = b.request();
  ASSERT(info.size > 0, "Invalid buffer");
  size_t result = 0;
  {
    pybind11::gil_scoped_release release_gil;
    result = pycodelet_node_->addTxBuffer(hook_tag, static_cast<size_t>(info.itemsize * info.size),
                                          info.ptr);
  }
  return result;
}

/*
 * As read from pybind11 code, pybind11 type wrappers access Python c-api for accessing
 * corresponding Py-instances, as shown below:
 * m_ptr = PyLong_FromLong((long) value);
 * These c-api talks to Python runtime.
 * As documented in Python c-api: https://docs.python.org/3/c-api/init.html
 * "The Python interpreter is not fully thread-safe. In order to support multi-threaded Python
 * programs, there’s a global lock, called the global interpreter lock or GIL, that must be held by
 * the current thread before it can safely access Python objects."
 * So the pybind11 code that manipulates python objects has to be protected by GIL.
 */

void InitPybindPyCodelet(pybind11::module& m) {
  pybind11::class_<isaac::alice::PybindPyCodelet>(m, "PybindPyCodelet")
      .def(pybind11::init<isaac::alice::PybindNode>())
      .def("python_wait_for_job", &isaac::alice::PybindPyCodelet::pythonWaitForJob)
      .def("python_job_finished", &isaac::alice::PybindPyCodelet::pythonJobFinished,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("tick_periodically", &isaac::alice::PybindPyCodelet::tickPeriodically)
      .def("tick_blocking", &isaac::alice::PybindPyCodelet::tickBlocking,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("tick_on_message", &isaac::alice::PybindPyCodelet::tickOnMessage)
      .def("synchronize", &isaac::alice::PybindPyCodelet::synchronize)
      .def("get_tick_time", &isaac::alice::PybindPyCodelet::getTickTime)
      .def("get_tick_dt", &isaac::alice::PybindPyCodelet::getTickDt)
      .def("is_first_tick", &isaac::alice::PybindPyCodelet::isFirstTick)
      .def("get_tick_count", &isaac::alice::PybindPyCodelet::getTickCount)
      .def("add_rx_hook", &isaac::alice::PybindPyCodelet::addRxHook)
      .def("add_tx_hook", &isaac::alice::PybindPyCodelet::addTxHook)
      .def("get_rx_buffer_content", &isaac::alice::PybindPyCodelet::getRxBufferContent)
      .def("available", &isaac::alice::PybindPyCodelet::available)
      .def("get_rx_pubtime", &isaac::alice::PybindPyCodelet::getRxPubtime)
      .def("get_rx_acqtime", &isaac::alice::PybindPyCodelet::getRxAcqtime)
      .def("get_rx_uuid", &isaac::alice::PybindPyCodelet::getRxUuid)
      .def("set_config", &isaac::alice::PybindPyCodelet::setConfig)
      .def("get_config", &isaac::alice::PybindPyCodelet::getConfig)
      .def("receive", &isaac::alice::PybindPyCodelet::receive)
      .def("publish", &isaac::alice::PybindPyCodelet::publish, pybind11::arg("message"),
           pybind11::arg("tag"), pybind11::arg("acqtime") = nullptr)  // acqtime is default to None
      .def("add_tx_buffer", &isaac::alice::PybindPyCodelet::addTxBuffer)
      .def("show", &isaac::alice::PybindPyCodelet::show);
}

}  // namespace alice
}  // namespace isaac
