/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "engine/gems/ml/object_detection_utils.hpp"

#include "engine/gems/geometry/n_cuboid.hpp"
#include "gtest/gtest.h"

namespace isaac {

// Tests the ComputeOverlap1d function with different combinations of overlapping bounding boxes
TEST(Utils, ComputeOverlap1d) {
  EXPECT_EQ(ComputeOverlap1d(5, 10, 15, 20), 0);
  EXPECT_EQ(ComputeOverlap1d(5, 17, 15, 20), 2);
  EXPECT_EQ(ComputeOverlap1d(15, 20, 5, 10), 0);
  EXPECT_EQ(ComputeOverlap1d(15, 20, 5, 17), 2);
}

// Tests the ComputeOverlap function which computes the overlap area between 2 bounding boxes
TEST(Utils, ComputeOverlap) {
  geometry::Rectanglef bounding_box1 =
      geometry::Rectanglef::FromOppositeCorners(Vector2f(4, 4), Vector2f(8, 8));
  geometry::Rectanglef bounding_box2 =
      geometry::Rectanglef::FromOppositeCorners(Vector2f(5, 5), Vector2f(10, 10));
  EXPECT_EQ(ComputeOverlap(bounding_box1, bounding_box2), 9);
  geometry::Boxf bounding_box3 =
      geometry::Boxf::FromOppositeCorners(Vector3f(4, 4, 4), Vector3f(8, 8, 8));
  geometry::Boxf bounding_box4 =
      geometry::Boxf::FromOppositeCorners(Vector3f(5, 5, 5), Vector3f(10, 10, 10));
  EXPECT_EQ(ComputeOverlap(bounding_box3, bounding_box4), 27);
}

// Tests the IntersectionOverUnion function to compuite the intersection over union of 2 bounding
// boxes
TEST(Utils, IntersectionOverUnion) {
  geometry::Rectanglef bounding_box1 =
      geometry::Rectanglef::FromOppositeCorners(Vector2f(4.0f, 4.0f), Vector2f(8.0f, 8.0f));
  geometry::Rectanglef bounding_box2 =
      geometry::Rectanglef::FromOppositeCorners(Vector2f(5.0f, 5.0f), Vector2f(10.0f, 10.0f));
  EXPECT_NEAR(IntersectionOverUnion(bounding_box1, bounding_box2), 0.281f, 0.001f);
  geometry::Boxf bounding_box3 =
      geometry::Boxf::FromOppositeCorners(Vector3f(4, 4, 4), Vector3f(8, 8, 8));
  geometry::Boxf bounding_box4 =
      geometry::Boxf::FromOppositeCorners(Vector3f(5, 5, 5), Vector3f(10, 10, 10));
  EXPECT_NEAR(IntersectionOverUnion(bounding_box3, bounding_box4), 0.166f, 0.001f);
}

// Tests the NonMaximumSuppression function with 3 bounding boxes
TEST(Utils, NonMaximumSuppression) {
  BoundingBoxDetection bounding_box_detection1, bounding_box_detection2, bounding_box_detection3;
  bounding_box_detection1.bounding_box =
      geometry::Rectangled::FromOppositeCorners({114, 66}, {178, 130});
  bounding_box_detection2.bounding_box =
      geometry::Rectangled::FromOppositeCorners({120, 60}, {184, 124});
  bounding_box_detection3.bounding_box =
      geometry::Rectangled::FromOppositeCorners({114, 60}, {178, 124});
  bounding_box_detection1.probability = 0.7;
  bounding_box_detection2.probability = 0.8;
  bounding_box_detection3.probability = 0.9;
  std::vector<BoundingBoxDetection> bounding_box_list = {
      bounding_box_detection1, bounding_box_detection2, bounding_box_detection3};
  BoundingBoxDetection filtered_box = NonMaximumSuppression(0.3, bounding_box_list)[0];
  geometry::Rectangled expected_bounding_box =
      geometry::Rectangled::FromOppositeCorners({114, 60}, {178, 124});
  EXPECT_NEAR(filtered_box.bounding_box.min().x(), expected_bounding_box.min().x(), 0.0001);
  EXPECT_NEAR(filtered_box.bounding_box.min().y(), expected_bounding_box.min().y(), 0.0001);
  EXPECT_NEAR(filtered_box.bounding_box.max().x(), expected_bounding_box.max().x(), 0.0001);
  EXPECT_NEAR(filtered_box.bounding_box.max().y(), expected_bounding_box.max().y(), 0.0001);
}

// Tests the Resize bounding box function to resize bounding boxes at {416, 416} resolution to
// {1280, 720} resolution.
TEST(Utils, ResizeBoundingBox) {
  geometry::Rectangled input_bounding_box =
      geometry::Rectangled::FromOppositeCorners({193.33, 140.25}, {246.14, 265.63});
  geometry::Rectangled output_bounding_box =
      ResizeBoundingBox(input_bounding_box, {416, 416}, {1328, 768}, 0);

  EXPECT_NEAR(output_bounding_box.min().x(), 617.16, 0.01);
  EXPECT_NEAR(output_bounding_box.min().y(), 167.72, 0.01);
  EXPECT_NEAR(output_bounding_box.max().x(), 785.75, 0.01);
  EXPECT_NEAR(output_bounding_box.max().y(), 567.97, 0.01);
}
}  // namespace isaac
