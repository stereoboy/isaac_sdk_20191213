/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "engine/alice/backend/stopwatch.hpp"
#include "engine/alice/backend/synchronization.hpp"
#include "engine/alice/component.hpp"
#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

class CodeletBackend;
class ChannelSynchronizer;

// Executes a custome piece of code
class Codelet : public Component {
 public:
  virtual ~Codelet() {}

  // The tick function is invoked whenever the codelet is expected to do work,
  // e.g., when an event is received, the period timing is reached.
  virtual void tick() {}

  // Tick as fast as possible (assuming that the tick uses blocking calls)
  void tickBlocking();
  // Tick periodically with given interval read from config. See tick_period
  void tickPeriodically();
  // Legacy tickPeriodically interface.  Deprecated.
  void tickPeriodically(double interval);
  // Tick each time a new messages is received by the given receiver
  void tickOnMessage(const RxMessageHook& rx);
  // Ticks when an event happens
  void tickOnEvents(const std::unordered_set<std::string>& events);
  // Synchronizes channels so that messages are only received when timestamps match. This function
  // can be called multiple times to synchronize multiple channels.
  void synchronize(const RxMessageHook& rx1, const RxMessageHook& rx2);
  // Converts the tick unit from string to seconds
  std::optional<double> getTickPeriodAsSeconds();
  // Converts the tick unit from string to Hz.
  std::optional<double> getTickPeriodAsFrequency();

  // Time at which the current tick started
  double getTickTime() const { return tick_data_.time; }
  // Time duration between the start of the current and the previous tick
  double getTickDt() const { return tick_data_.dt; }
  // Tick at which current tick started
  int64_t getTickTimestamp() const { return tick_data_.timestamp; }

  // Returns true if this is the first tick after start
  bool isFirstTick() const { return tick_data_.count == 1; }
  // Returns the number of times a codelet ticked
  size_t getTickCount() const { return tick_data_.count; }
  // Returns a stopwatch for the given nameSleep tag.
  Stopwatch& stopwatch(const std::string& clock_name = "");

  // Helper function to show a variable with sight
  template <typename T, std::enable_if_t<std::is_arithmetic<std::decay_t<T>>::value, int> = 0>
  void show(const std::string& tag, T value) const {
    node()->sight().show(this, tag, this->tick_data_.timestamp, value);
  }
  // Helper function to show a variable with sight
  template <typename T, std::enable_if_t<std::is_arithmetic<std::decay_t<T>>::value, int> = 0>
  void show(const std::string& tag, int64_t timestamp, T value) const {
    node()->sight().show(this, tag, timestamp, value);
  }
  // Helper function to show everything except a variable with sight
  template <typename T, std::enable_if_t<!std::is_arithmetic<std::decay_t<T>>::value, int> = 0>
  void show(const std::string& tag, T&& arg) const {
    node()->sight().show(this, tag, getTickTime(), std::forward<T>(arg));
  }
  // Helper function to show everything except a variable with sight
  template <typename T, std::enable_if_t<!std::is_arithmetic<std::decay_t<T>>::value, int> = 0>
  void show(const std::string& tag, double time, T&& arg) const {
    node()->sight().show(this, tag, time, std::forward<T>(arg));
  }

  // Helper function to set a variable
  void setVariable(const std::string& tag, double value) const {
    setVariable(tag, this->tick_data_.timestamp, value);
  }
  void setVariable(const std::string& tag, int64_t timestamp, double value) const;
  // Helper function to get a variable
  std::optional<double> getVariable(const std::string& link) const {
    return getVariable(link, this->tick_data_.timestamp);
  }
  std::optional<double> getVariable(const std::string& link, int64_t timestamp) const;

  // Internal usage // TODO: protect this function better
  void addRx(RxMessageHook* rx) {
    rx_hook_trackers_[rx->tag()] = {rx};
  }

  // Config parameter for setting tick period.  Units support are s, ms, and hz
  // If no unit is specified seconds are assumed.
  ISAAC_PARAM(std::string, tick_period);

  // In case the codelet is triggered while already running it is queued at most the given number
  // of times. If set to -1 no limit will be applied. Using no limit is discouraged as it can lead
  // to queues growing without bound and thus unbounded memory usage. By default the codelet
  // can be queued at most once. This allows it to fire again immediately after it is done with
  // execution to react to data which arrived while it was processing previous data.
  ISAAC_PARAM(int, execution_queue_limit, 1);

 private:
  friend class CodeletBackend;

  // Struct for trackining timing information for the codelet
  struct Tick {
    // Time of the current tick in seconds
    double time;
    // Time of the previous tick in seconds
    double time_previous;
    // Tick delta in seconds
    double dt;
    // Timestamp of the current tick in ticks
    int64_t timestamp;
    // Number of times the codelet has ticked
    size_t count;
  };

  // To be called before the codelet starts
  void onBeforeTick();
  // To be called before the codelet ticks
  void onBeforeStart();
  // To be called before the codelet stops
  void onBeforeStop();

  void notifyBackendThatTickingChanged();
  std::optional<double> convertTimeUnitToSeconds();

  CodeletBackend* backend_ = nullptr;

  int64_t tick_period_ = -1;
  std::unordered_set<std::string> triggers_;
  bool non_rx_triggered_ = false;

  std::unordered_map<std::string, Stopwatch> stopwatches_;

  struct RxHookTracker {
    RxMessageHook* rx = nullptr;
    std::optional<int64_t> last_timestamp;
  };

  std::map<std::string, RxHookTracker> rx_hook_trackers_;
  std::set<std::unique_ptr<ChannelSynchronizer>> synchronizers_;
  Tick tick_data_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT_BASE(isaac::alice::Codelet)
