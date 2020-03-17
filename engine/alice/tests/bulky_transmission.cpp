/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "bulky_transmission.hpp"

#include <string>

#include "gtest/gtest.h"

namespace isaac {
namespace alice {

void BulkyTransmitter::start() {
  tickPeriodically();
}

void BulkyTransmitter::tick() {
  const int chunk_size = get_chunk_size();
  auto outmsg = tx_out().initProto();
  outmsg.setFoo(43);
  outmsg.setBar("hello");
  auto data1 = outmsg.initData1(chunk_size);
  for (size_t i = 0; i < data1.size(); i++) {
    data1[i] = i%19;
  }
  auto data2 = outmsg.initData2(3*chunk_size);
  for (size_t i = 0; i < data2.size(); i++) {
    data2[i] = i%29;
  }
  auto data3 = outmsg.initData3(5*chunk_size);
  for (size_t i = 0; i < data3.size(); i++) {
    data3[i] = i%39;
  }
  auto data4 = outmsg.initData4(7*chunk_size);
  for (size_t i = 0; i < data4.size(); i++) {
    data4[i] = i%49;
  }
  tx_out().publish();
}

void BulkyTransmitter::stop() {
  EXPECT_GT(getTickCount(), 0);
}

void BulkyReceiver::start() {
  tickOnMessage(rx_in());
}

void BulkyReceiver::tick() {
  const int chunk_size = get_chunk_size();
  EXPECT_EQ(uuids_.count(rx_in().message_uuid()), 0);
  uuids_.insert(rx_in().message_uuid());
  auto msg = rx_in().getProto();
  EXPECT_EQ(msg.getFoo(), 43);
  std::string str = msg.getBar();
  EXPECT_STREQ(str.c_str(), "hello");
  auto data1 = msg.getData1();
  ASSERT_EQ(msg.getData1().size(), chunk_size);
  for (size_t i = 0; i < data1.size(); i++) {
    EXPECT_EQ(data1[i], i%19);
  }
  auto data2 = msg.getData2();
  ASSERT_EQ(data2.size(), 3*chunk_size);
  for (size_t i = 0; i < data2.size(); i++) {
    EXPECT_EQ(data2[i], i%29);
  }
  auto data3 = msg.getData3();
  ASSERT_EQ(data3.size(), 5*chunk_size);
  for (size_t i = 0; i < data3.size(); i++) {
    EXPECT_EQ(data3[i], i%39);
  }
  auto data4 = msg.getData4();
  ASSERT_EQ(data4.size(), 7*chunk_size);
  for (size_t i = 0; i < data4.size(); i++) {
    EXPECT_EQ(data4[i], i%49);
  }
}

void BulkyReceiver::stop() {
  EXPECT_GT(getTickCount(), 0);
}

}  // namespace alice
}  // namespace isaac
