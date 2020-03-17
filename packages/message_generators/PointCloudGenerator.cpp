/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include "PointCloudGenerator.hpp"

#include <algorithm>
#include <utility>
#include <vector>

namespace isaac {
namespace message_generators {

void PointCloudGenerator::start() {
  // Prime the number of points remaining to be sent per the parameter.
  remaining_points_ = get_point_count();

  tickPeriodically();
}

void PointCloudGenerator::tick() {
  // Send a batch per tick, as long as we have remaining points to send.
  sendBatch();
}

void PointCloudGenerator::sendBatch() {
  // Handle reamining count and ensure we don't send more than we should.
  if (remaining_points_ > 0) {
    size_t points_to_send = std::min(remaining_points_, get_point_per_message());

    // Initialize and populate the message per test requirements.
    auto proto = tx_point_cloud().initProto();
    proto.initPositions();
    std::vector<SharedBuffer>& buffers = tx_point_cloud().buffers();

    SampleCloud3f positions(points_to_send);
    for (size_t point_index = 0; point_index < points_to_send; point_index++) {
      // Generate a wavy surface as test data.
      positions[point_index][0] = point_index % 100;
      positions[point_index][1] = point_index / 100;
      positions[point_index][2] = std::cos((point_index % 100) / 30.0) * 10.0;
    }

    if (get_has_normals()) {
      proto.initNormals();
      SampleCloud3f normals(points_to_send);
      for (size_t point_index = 0; point_index < points_to_send; point_index++) {
        // Generate normals as normalized point world position.
        normals[point_index] = positions[point_index].normalized();
      }
      ToProto(std::move(normals), proto.getNormals(), buffers);
    }

    if (get_has_colors()) {
      proto.initColors();
      SampleCloud3f colors(points_to_send);
      for (size_t point_index = 0; point_index < points_to_send; point_index++) {
        // Generate colors for each point.
        float tint = std::cos((point_index % 100) / 30.0) / 2.0 + 0.5;
        Vector3f color{tint, tint, tint};
        colors[point_index] = color;
      }
      ToProto(std::move(colors), proto.getColors(), buffers);
    }

    if (get_has_intensities()) {
      proto.initIntensities();
      SampleCloud1f intensities(points_to_send);
      for (size_t point_index = 0; point_index < points_to_send; point_index++) {
        // Generate an intensity for each point.
        intensities[point_index][0] = std::sin((point_index % 100) / 30.0) / 2.0 + 0.5;
      }
      ToProto(std::move(intensities), proto.getIntensities(), buffers);
    }

    ToProto(std::move(positions), proto.getPositions(), buffers);
    tx_point_cloud().publish();
    remaining_points_ -= points_to_send;
  }
  // Else: We do nothing if there is no more data to send.
}

}  // namespace message_generators
}  // namespace isaac
