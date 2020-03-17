/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/gems/geometry/n_cuboid.hpp"
#include "engine/gems/geometry/plane.hpp"
#include "messages/geometry.capnp.h"
#include "messages/math.hpp"

namespace isaac {

inline geometry::PlaneD FromProto(::PlaneProto::Reader reader) {
  return {FromProto(reader.getNormal()), reader.getOffset()};
}

inline void ToProto(const geometry::PlaneD& plane, ::PlaneProto::Builder builder) {
  ToProto(plane.normal(), builder.initNormal());
  builder.setOffset(plane.offset());
}

inline geometry::Rectangled FromProto(::RectangleProto::Reader reader) {
  return geometry::Rectangled::FromOppositeCorners(FromProto(reader.getMin()),
                                                   FromProto(reader.getMax()));
}

inline void ToProto(const geometry::Rectangled& rectangle, ::RectangleProto::Builder builder) {
  ToProto(rectangle.min(), builder.initMin());
  ToProto(rectangle.max(), builder.initMax());
}

}  // namespace isaac
