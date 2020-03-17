/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "image_transmission.hpp"

#include <utility>

#include "gtest/gtest.h"

namespace isaac {
namespace alice {

void ImageTransmitter::start() {
  tickPeriodically();
}

void ImageTransmitter::tick() {
  Image1ub image(get_rows(), get_cols());
  for (size_t i = 0; i < image.rows(); i++) {
    for (size_t j = 0; j < image.cols(); j++) {
      image(i, j) = (i * image.cols() + j) % 256;
    }
  }
  ToProto(std::move(image), tx_image().initProto(), tx_image().buffers());
  tx_image().publish();
}

void ImageTransmitter::stop() {
  EXPECT_GT(getTickCount(), 0);
}

void ImageReceiver::start() {
  tickOnMessage(rx_image());
}

void ImageReceiver::tick() {
  ImageConstView1ub image;
  const bool ok = FromProto(rx_image().getProto(), rx_image().buffers(), image);
  ASSERT_TRUE(ok);
  ASSERT_EQ(image.rows(), get_rows());
  ASSERT_EQ(image.cols(), get_cols());
  for (size_t i = 0; i < image.rows(); i++) {
    for (size_t j = 0; j < image.cols(); j++) {
      ASSERT_EQ(image(i, j), (i * image.cols() + j) % 256);
    }
  }
}

void ImageReceiver::stop() {
  EXPECT_GT(getTickCount(), 0);
}

}  // namespace alice
}  // namespace isaac
