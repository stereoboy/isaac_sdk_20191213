/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "engine/core/math/types.hpp"
#include "engine/gems/geometry/line_segment.hpp"
#include "engine/gems/geometry/line_utils.hpp"

namespace isaac {
namespace geometry {

// A polygon in 2D, the last point and first point are connected
template <typename K>
struct Polygon2 {
  // Type of a point.
  using Vector_t = Vector2<K>;
  using Scalar = K;

  std::vector<Vector_t> points;

  // Returns whether a point is inside the polygon
  // Note in case the point lies on the same points as the polygon, it will return false
  bool isInside(const Vector_t& pt) const {
    // Select a point such as the segment [pt, max] will not intersect with any point of the polygon
    // and such as max is outside the polygon
    Vector_t max(pt.x(), pt.y() + K(1));
    for (const auto& point : points) {
      max.x() = std::max(max.x(), point.x() + K(1));
      if (point.y() > pt.y() && point.y() < max.y()) {
        max.y() = point.y();
      }
    }
    max.y() -= (max.y() - pt.y()) * K(0.5);
    const LineSegment<K, 2> seg(pt, max);
    // Count how many segment it intersects, if it's an odd number then the point is inside or on
    // one edge.
    int counter = 0;
    for (size_t ix = 0; ix < points.size(); ix++) {
      const LineSegment<K, 2> edge(points[ix], points[(ix + 1) % points.size()]);
      if (AreLinesIntersecting(seg, edge)) {
        counter++;
      }
    }
    return counter % 2 == 1;
  }

  // Returns the distance of a point from the polygon edges
  K distance(const Vector_t& pt, Vector_t* grad = nullptr) const {
    K squared_dist = std::numeric_limits<K>::max();
    for (size_t ix = 0; ix < points.size(); ix++) {
      const LineSegment<K, 2> edge(points[ix], points[(ix + 1) % points.size()]);
      const Vector_t closest = ClosestPointToLine(edge, pt);
      const K dist = (pt - closest).squaredNorm();
      if (dist < squared_dist) {
        squared_dist = dist;
        if (grad) *grad = pt - closest;
      }
    }
    if (grad) grad->normalize();
    return std::sqrt(squared_dist);
  }

  // Returns the signed distance of a point from the polygon (negative means inside the polygon)
  K signedDistance(const Vector_t& pt, Vector_t* grad = nullptr) const {
    const K dist = distance(pt, grad);
    // If the point is inside the polygon, we need to revert the direction of the gradient.
    if (isInside(pt)) {
      if (grad) *grad = -(*grad);
      return -dist;
    }
    return dist;
  }

  // Casts to a different type
  template <typename S, typename std::enable_if_t<!std::is_same<S, K>::value, int> = 0>
  Polygon2<S> cast() const {
    std::vector<Vector2<S>> pts;
    pts.reserve(points.size());
    for (const auto& pt : points) {
      pts.emplace_back(pt.template cast<S>());
    }
    return Polygon2<S>{std::move(pts)};
  }
  template<typename S, typename std::enable_if_t<std::is_same<S, K>::value, int> = 0>
  const Polygon2& cast() const {
    // Nothing to do as the type does not change
    return *this;
  }
};

// A polygon in 3D, the last point and first point are connected
template <typename K>
struct Polygon3 {
  // Type of a point.
  using Vector_t = Vector3<K>;
  using Scalar = K;

  std::vector<Vector_t> points;
};

using Polygon2D = Polygon2<double>;
using Polygon2F = Polygon2<float>;
using Polygon3D = Polygon3<double>;
using Polygon3F = Polygon3<float>;

}  // namespace geometry
}  // namespace isaac
