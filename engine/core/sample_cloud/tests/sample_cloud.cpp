/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/core/sample_cloud/sample_cloud.hpp"

#include "engine/core/math/types.hpp"
#include "gtest/gtest.h"

namespace isaac {

TEST(SampleCloud, Creation) {
  SampleCloud2d pc1;
  EXPECT_EQ(pc1.channels(), 2);
  EXPECT_EQ(pc1.size(), 0);
  EXPECT_EQ(pc1.data().size(), 0);
  SampleCloud2f pc2(10);
  EXPECT_EQ(pc2.channels(), 2);
  EXPECT_EQ(pc2.size(), 10);
  EXPECT_EQ(pc2.data().size(), 80);
  SampleCloud4i pc3(100);
  EXPECT_EQ(pc3.channels(), 4);
  EXPECT_EQ(pc3.size(), 100);
  EXPECT_EQ(pc3.data().size(), 1600);
}

TEST(SampleCloud, Views) {
  SampleCloud2d pc1(100);
  SampleCloudView2d pcv1 = pc1.view();
  EXPECT_EQ(pcv1.size(), 100);
  EXPECT_EQ(pcv1.data().size(), pc1.data().size());
  EXPECT_EQ(pcv1.data().begin(), pc1.data().begin());

  SampleCloud2f pc2(10);
  SampleCloudConstView2f pcv2 = pc2.const_view();
  EXPECT_EQ(pcv2.size(), 10);
  EXPECT_EQ(pcv2.data().size(), pc2.data().size());
  EXPECT_EQ(pcv2.data().begin(), pc2.data().begin());
}

TEST(SampleCloud, SetAndGetInterleaved) {
  SampleCloud3d pc(10);
  for (size_t i = 0; i < pc.size(); ++i) {
    pc[i] = Vector3d{static_cast<double>(i),
                     static_cast<double>(i + 1),
                     static_cast<double>(i + 2)};
  }
  for (size_t i = 0; i < pc.size(); ++i) {
    auto data = pc[i];
    ASSERT_EQ(data[0], static_cast<double>(i));
    ASSERT_EQ(data[1], static_cast<double>(i + 1));
    ASSERT_EQ(data[2], static_cast<double>(i + 2));
  }
}

TEST(SampleCloud, EigenView) {
  {
    SampleCloud3d pc(200);
    for (size_t i = 0; i < pc.size(); ++i) {
      pc[i] = Vector3d{static_cast<double>(i),
                       static_cast<double>(i + 1),
                       static_cast<double>(i + 2)};
    }
    auto data = pc.eigen_const_view();
    for (size_t i = 0; i < pc.size(); ++i) {
      ASSERT_EQ(data(0, i), static_cast<double>(i));
      ASSERT_EQ(data(1, i), static_cast<double>(i + 1));
      ASSERT_EQ(data(2, i), static_cast<double>(i + 2));
    }
  }
  {
    SampleCloud3d pc(200);
    auto view = pc.eigen_view();
    for (size_t i = 0; i < pc.size(); ++i) {
      view(0, i) = static_cast<double>(i);
      view(1, i) = static_cast<double>(i + 1);
      view(2, i) = static_cast<double>(i + 2);
    }
    auto data = pc.eigen_const_view();
    for (size_t i = 0; i < pc.size(); ++i) {
      ASSERT_EQ(data(0, i), static_cast<double>(i));
      ASSERT_EQ(data(1, i), static_cast<double>(i + 1));
      ASSERT_EQ(data(2, i), static_cast<double>(i + 2));
    }
  }
}

}  // namespace isaac
