/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <limits>

#include "engine/core/math/utils.hpp"

namespace isaac {

// Simple PID controller using 3 gain
template <typename K>
class PidController {
 public:
  static_assert(std::is_floating_point<K>::value, "K must be floating point type");
  // Parameters specific to the problem.
  struct Parameters {
    K propotional_gain = 1.0;
    K integral_gain = 1.0;
    K derivative_gain = 1.0;

    K max_output = std::numeric_limits<K>::max();
    K min_output = -std::numeric_limits<K>::max();

    K max_integral = std::numeric_limits<K>::max();
    K min_integral = -std::numeric_limits<K>::max();
  };

  // Computes the output given the target (setpoint), the current value, and the delta time since
  // the last execution.
  K calculate(K setpoint, K current_value, K dt) {
    // Make sure dt > 0, otherwise return the last output.
    if (dt <= K(0)) return last_output_;
    const K error = setpoint - current_value;
    // If this is the first time we run it, we need to initialize the last_error_
    if (first_run_) {
      first_run_ = false;
      last_error_ = error;
    }

    // Compute the proportional term
    const K p_out = parameters.propotional_gain * error;

    // Update the integral
    integral_ += error * dt;
    // Clamp the integral if needed
    integral_ = Clamp(integral_, parameters.min_integral, parameters.max_integral);

    // Compute the integral term
    const K i_out = parameters.integral_gain * integral_;

    // Derivative term
    const K derivative = (error - last_error_) / dt;
    const K d_out = parameters.derivative_gain * derivative;

    // Calculate total output
    K output = p_out + i_out + d_out;

    // Clamp the output if needed
    output = Clamp(output, parameters.min_output, parameters.max_output);
    // Save error to previous error
    last_error_ = error;
    last_output_ = output;

    return output;
  }

  Parameters parameters;

 private:
  bool first_run_ = true;
  K integral_ = 0.0;
  K last_error_ = 0.0;
  K last_output_ = 0.0;
};

}  // namespace isaac
