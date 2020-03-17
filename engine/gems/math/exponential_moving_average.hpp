/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cmath>

namespace isaac {
namespace math {

// Helper function to maintain a moving average of an observation. It helps reduce noise in
// measurements.
// Usage:
//  ExponentialMovingAverage(time_window)
//  add(new_observation, time) -> returns the current best estimaton.
template <class K>
class ExponentialMovingAverage {
 public:
  ExponentialMovingAverage() : ExponentialMovingAverage(K(1)) {}
  ExponentialMovingAverage(K lamda)
  : lamda_inv_(K(1) / lamda), is_first_time_(true), current_time_(K(0)), current_value_(K(0)) {}

  // Adds a new measurement, updates the current value and returns it.
  K add(K value, K time) {
    if (is_first_time_) {
      current_value_ = value;
      current_time_ = time;
      is_first_time_ = false;
      return value;
    }
    const K dt = time - current_time_;
    // We receive a previous measurement, unfortunately we can't incorporate it.
    if (dt <= 0) return current_value_;
    const K w = K(1) - std::exp(-dt * lamda_inv_);
    current_value_ -= w * (current_value_ - value);
    current_time_ = time;
    return current_value_;
  }

  // Returns the last time we have got an update
  K time() const {
    return current_time_;
  }

  // Returns the current averaged value.
  K value() const {
    return current_value_;
  }

 protected:
  K lamda_inv_;
  bool is_first_time_;
  K current_time_;
  K current_value_;
};

// Helper function to maintain a moving average of a rate.
// Usage:
//  ExponentialMovingAverageRate(time_window)
//  add(additive_flow, time) -> returns the current best estimaton.
template <class K>
class ExponentialMovingAverageRate {
 public:
  ExponentialMovingAverageRate() : ExponentialMovingAverageRate(K(1)) {}
  ExponentialMovingAverageRate(K lamda)
  : lamda_inv_(K(1) / lamda), is_first_time_(true), current_time_(K(0)), current_rate_(K(0)) {}

  // Adds a new measurement, updates the current value and returns it.
  K add(K flow, K time) {
    if (is_first_time_) {
      current_time_ = time;
      current_rate_ = flow * lamda_inv_;
      is_first_time_ = false;
      return current_rate_;
    }
    const K dt = time - current_time_;
    if (dt <= 0) {
      // We receive a previous measurement, let's just add it.
      current_rate_ += flow * lamda_inv_;
    } else {
      // Decay the rate exponentially
      current_rate_ -= adjustmentFactor(dt) * (dt * current_rate_ - flow);
      current_time_ = time;
    }
    return current_rate_;
  }

  // Updates the last time (decays the current value accordingly).
  void updateTime(K time) {
    const K dt = time - current_time_;
    if (dt <= 0) return;
    current_rate_ *= decayFactor(dt);
    current_time_ = time;
  }

  // Returns the last time we have got an update
  K time() const { return current_time_; }

  // Returns the current estimated rate.
  K rate() const { return current_rate_; }

 protected:
  // Helper function computing the second-order approximation of the function `(1 - exp(-x)) / x`.
  static K Approximation(K x) {
    return K(1) + x * (x * K(1.0 / 6.0) - K(0.5));
  }

  // Decay factor for given time period computed as `exp(-dt/l)`.
  // If dt is small enough we can use the second order approximation.
  K decayFactor(K dt) const {
    const K dx = dt * lamda_inv_;
    return (dx < K(0.1)) ? (K(1) - dx * Approximation(dx)) : std::exp(-dx);
  }

  // Rate adjustment factor for given time period computed as `(1 - exp(dt/l)) / dt`.
  // If dt is small enough we can use the third order approximation.
  K adjustmentFactor(K dt) const {
    const K dx = dt * lamda_inv_;
    return (dx < K(0.1)) ? (Approximation(dx) * lamda_inv_) : ((K(1) - std::exp(-dx)) / dt);
  }

  K lamda_inv_;
  bool is_first_time_;
  K current_time_;
  K current_rate_;
};

}  // namespace math
}  // namespace isaac
