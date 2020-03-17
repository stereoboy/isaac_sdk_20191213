/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Codelet.hpp"

#include <algorithm>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/any_storage.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/codelet_backend.hpp"
#include "engine/alice/backend/synchronization.hpp"
#include "engine/core/assert.hpp"
#include "engine/core/time.hpp"

namespace isaac {
namespace alice {

void Codelet::tickBlocking() {
  tick_period_ = 0;
  triggers_.clear();
  non_rx_triggered_ = false;
  notifyBackendThatTickingChanged();
}

std::optional<double> Codelet::convertTimeUnitToSeconds() {
  std::string param = get_tick_period();
  // Convert string to standard lower form.
  std::transform(param.begin(), param.end(), param.begin(), ::tolower);
  // Check unit type. Assume seconds unless we find hz or ms.
  bool isFrequency = false;
  double timeScale = 1.0;
  const size_t idx_hz = param.find("hz");
  const size_t idx_ms = param.find("ms");

  if (idx_hz != std::string::npos && idx_ms != std::string::npos) {
    LOG_ERROR("Invalid tick period");
    return std::nullopt;
  }
  if (idx_hz != std::string::npos) {
    isFrequency = true;
  }
  // If unit is ms needto scale the final result by dividing by 1000
  if (idx_ms != std::string::npos) {
    timeScale = .001;
  }
  errno = 0;
  double paramVal = std::strtod(param.c_str(), nullptr);

  if (errno != 0 || !std::isfinite(paramVal) || paramVal <= 0.0) {
    LOG_ERROR("Invalid tick period for %s", full_name().c_str());
    return std::nullopt;
  }
  errno = 0;
  // If unit is Hz need to convert final result to seconds by inverting the value.
  if (isFrequency) {
    paramVal = 1.0 / paramVal;
  } else {
    paramVal = paramVal * timeScale;
  }
  return std::optional<double>(paramVal);
}

void Codelet::tickPeriodically() {
  auto time = convertTimeUnitToSeconds();
  if (!time) {
    LOG_ERROR("Unable to tick due to invalid tick period");
    return;
  }
  tick_period_ = SecondsToNano(time.value());
  triggers_.clear();
  non_rx_triggered_ = false;
  notifyBackendThatTickingChanged();
}

void Codelet::tickPeriodically(double interval) {
  LOG_WARNING("Function deprecated. Set tick_period to the desired tick paramater");
  set_tick_period(std::to_string(interval));
  tickPeriodically();
}

std::optional<double> Codelet::getTickPeriodAsSeconds() {
  return convertTimeUnitToSeconds();
}

std::optional<double> Codelet::getTickPeriodAsFrequency() {
  auto time = convertTimeUnitToSeconds();
  if (time) {
    return std::optional<double>(1.0 / time.value());
  } else {
    return std::nullopt;
  }
}

void Codelet::tickOnMessage(const RxMessageHook& rx) {
  ASSERT(rx.component() == this, "Can not tick on a message hook from another codelet");
  triggers_.insert(rx.channel_id());
  tick_period_ = -1;
  notifyBackendThatTickingChanged();
}

void Codelet::tickOnEvents(const std::unordered_set<std::string>& events) {
  tick_period_ = -1;
  triggers_ = events;
  non_rx_triggered_ = true;
  notifyBackendThatTickingChanged();
}

void Codelet::synchronize(const RxMessageHook& rx1, const RxMessageHook& rx2) {
  ASSERT(rx1.component() == this, "Can not synchronize with a message hook from another codelet");
  ASSERT(rx2.component() == this, "Can not synchronize with a message hook from another codelet");
  ASSERT(rx1.tag() != rx2.tag(), "Can not synchronize a channel with itself");
  ChannelSynchronizer* found_sync = nullptr;
  // If either of the two RX hooks are already part of a synchronizer group add to that group
  for (const auto& sync : synchronizers_) {
    if (sync->contains(rx1.tag())) {
      ASSERT(!sync->contains(rx2.tag()), "Channels are already synchronized");
      sync->mark(rx2.tag());
      found_sync = sync.get();
      break;
    }
    if (sync->contains(rx2.tag())) {
      ASSERT(!sync->contains(rx1.tag()), "Channels are already synchronized");
      sync->mark(rx1.tag());
      found_sync = sync.get();
      break;
    }
  }
  if (!found_sync) {
    // add a new one
    auto uptr = std::make_unique<ChannelSynchronizer>();
    uptr->mark(rx1.tag());
    uptr->mark(rx2.tag());
    synchronizers_.insert(std::move(uptr));
    found_sync = uptr.get();
  }
}

void Codelet::notifyBackendThatTickingChanged() {
  ASSERT(backend_ != nullptr,
         "Codelet not registered with backend. Did you called tickBlocking, tickPeriodically, or "
         "tickOnMessage in the constructor?");
  backend_->onChangeTicking(this);
}

void Codelet::onBeforeStart() {
  tick_data_.timestamp = node()->clock()->timestamp();
  tick_data_.time = ToSeconds(tick_data_.timestamp);
  tick_data_.time_previous = tick_data_.time;
  tick_data_.dt = 0.0;
  tick_data_.count = 0;
}

void Codelet::onBeforeTick() {
  tick_data_.timestamp = node()->clock()->timestamp();
  tick_data_.time_previous = tick_data_.time;
  tick_data_.time = ToSeconds(tick_data_.timestamp);
  tick_data_.dt = tick_data_.time - tick_data_.time_previous;
  tick_data_.count++;
}

void Codelet::onBeforeStop() {
  onBeforeTick();
  tick_data_.count--;  // correct the tick count
}

Stopwatch& Codelet::stopwatch(const std::string& name) {
  Stopwatch& stopwatch = stopwatches_[name];
  if (!stopwatch.valid()) {
    stopwatch.setClock(node()->clock());
  }
  return stopwatch;
}

void Codelet::setVariable(const std::string& tag, int64_t timestamp, double value) const {
  // FIXME use timestamp
  node()->app()->backend()->any_storage()->set(full_name() + "/" + tag, value);  // NOLINT
  show(tag, timestamp, value);
}

std::optional<double> Codelet::getVariable(const std::string& link, int64_t timestamp) const {
  // FIXME use timestamp
  return node()->app()->backend()->any_storage()->tryGet(link);
}

}  // namespace alice
}  // namespace isaac
