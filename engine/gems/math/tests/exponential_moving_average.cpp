/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/gems/math/exponential_moving_average.hpp"

#include <random>

#include "gtest/gtest.h"

namespace isaac {
namespace math {

namespace {
std::mt19937 rng;
}  // namespace

TEST(ExponentialMovingAverage, normal_usage) {
  ExponentialMovingAverage<double> ema(2.0);
  std::uniform_real_distribution<double> obs_dis(4.0, 6.0);
  std::uniform_real_distribution<double> time_dis(0.01, 0.1);
  // Initialize with some value (the beginning is expected to be noisy)
  double time = 0.0;
  while (time < 4.0) {
    time += time_dis(rng);
    ema.add(obs_dis(rng), time);
  }
  while (time < 50.0) {
    time += time_dis(rng);
    EXPECT_NEAR(ema.add(obs_dis(rng), time), 5.0, 0.3);
  }
}

TEST(ExponentialMovingAverageRate, normal_usage) {
  ExponentialMovingAverageRate<double> ema(2.0);
  std::normal_distribution<double> flow_dis(100.0, 10.0);
  std::uniform_real_distribution<double> time_dis(0.0, 0.1);
  // Initialize with some value (the beginning is expected to be noisy)
  double time = 0.0;
  while (time < 10.0) {
    const double dt = time_dis(rng);
    time += dt;
    ema.add(flow_dis(rng) * dt, time);
  }
  while (time < 50.0) {
    const double dt = time_dis(rng);
    time += dt;
    EXPECT_NEAR(ema.add(flow_dis(rng) * dt, time), 100.0, 5.0);
  }
  while (time < 120.0) {
    const double dt = time_dis(rng);
    time += dt;
    ema.updateTime(time);
  }
  EXPECT_NEAR(ema.rate(), 0.0, 1e-2);
}

TEST(ExponentialMovingAverageRate, low_frequency) {
  ExponentialMovingAverageRate<double> ema(2.0);
  std::normal_distribution<double> flow_dis(100.0, 5.0);
  std::uniform_real_distribution<double> time_dis(0.5, 1.5);
  // Initialize with some value (the beginning is expected to be noisy)
  double time = 0.0;
  while (time < 20.0) {
    const double dt = time_dis(rng);
    time += dt;
    ema.add(flow_dis(rng) * dt, time);
  }
  while (time < 100.0) {
    const double dt = time_dis(rng);
    time += dt;
    EXPECT_NEAR(ema.add(flow_dis(rng) * dt, time), 100.0, 6.0);
  }
  while (time < 120.0) {
    const double dt = time_dis(rng);
    time += dt;
    ema.updateTime(time);
  }
  EXPECT_NEAR(ema.rate(), 0.0, 1e-2);

}

}  // namespace math
}  // namespace isaac
