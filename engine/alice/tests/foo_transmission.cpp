/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "foo_transmission.hpp"

#include <string>

#include "gtest/gtest.h"

namespace isaac {
namespace alice {

namespace {
constexpr float kValue = 3.1415f;
constexpr char kText[] = "so long";
}

void FooTransmitter::start() {
  tickPeriodically();
}

void FooTransmitter::tick() {
  auto foo = tx_foo().initProto();
  foo.setCount(get_offset() + static_cast<int>(getTickCount()));
  foo.setValue(kValue);
  foo.setText(kText);
  tx_foo().publish();
}

void FooTransmitter::stop() {
  auto expected_tick_count = try_get_expected_tick_count();
  if (expected_tick_count) {
    EXPECT_NEAR(getTickCount(), *expected_tick_count, get_expected_tick_count_tolerance());
  }
}

void FooReceiver::start() {
  auto tick_period = try_get_tick_period();
  if (tick_period) {
    tickPeriodically();
    is_periodic_ = true;
  } else {
    tickOnMessage(rx_foo());
    is_periodic_ = false;
  }
}

void FooReceiver::tick() {
  tick_count = getTickCount();
  if (is_periodic_) {
    if (!rx_foo().available()) {
      return;
    }
  } else {
    ASSERT_TRUE(rx_foo().available());
  }
  const auto& foo = rx_foo().getProto();
  EXPECT_NEAR(foo.getCount(), get_offset() + static_cast<int>(getTickCount()),
              get_count_tolerance());
  EXPECT_EQ(foo.getValue(), kValue);
  EXPECT_EQ(foo.getText(), kText);
  if (on_tick_callback) {
    on_tick_callback(this);
  }
}

void FooReceiver::stop() {
  auto expected_tick_count = try_get_expected_tick_count();
  if (expected_tick_count) {
    EXPECT_NEAR(getTickCount(), *expected_tick_count, get_expected_tick_count_tolerance());
  }
}

}  // namespace alice
}  // namespace isaac
