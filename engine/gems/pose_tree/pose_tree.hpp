/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>
#include <set>
#include <utility>
#include <vector>

#include "engine/core/math/pose3.hpp"
#include "engine/core/optional.hpp"
#include "engine/gems/algorithm/timeseries.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace pose_tree {

// A temporal pose tree to store relative coordinate system transformations over time
// This implementation does not support multiple paths between the same coordinate systems. It does
// however allow for multiple "roots". In fact the transformation relationships form an acylic,
// bi-directional, not necessarily fully-connected graph.
class PoseTree {
 public:
  using PoseHistory = Timeseries<Pose3d, double>;

  // Checks if there is a direct connection between two coordinate systems at a given time
  bool hasDirectConnection(const Uuid& lhs, const Uuid& rhs, double stamp) const;
  // Checks if there is an indirect connection between two coordinate systems at a given time.
  // This means a lhs and rhs are not connected directly, but there is a sequence of direct
  // connections to get from lhs to rhs.
  bool hasIndirectConnection(const Uuid& lhs, const Uuid& rhs, double stamp) const;

  // Gets the 3D pose lhs_T_rhs at given time
  // This will try to find a path between the two specified coordinate frames. If there is no path
  // between the two coordinate frames this function will return std::nullopt.
  // TODO The current implementation will be slow for big graphs
  // TODO It uses the access policies of timeseries which will give first or last values if the
  // there is no element at the given time. This implies that if two coordinate frames where
  // connected once they currently remain connected forever.
  std::optional<Pose3d> get(const Uuid& lhs, const Uuid& rhs, double stamp) const;

  // Sets the 3D pose lhs_T_rhs (and rhs_T_lhs) at given time
  // If the connection would add a cycle to the graph this function will return false.
  bool set(const Uuid& lhs, const Uuid& rhs, double stamp, const Pose3d& lhs_T_rhs);

  // Gets the transformation a_t1_T_a_t2 using `base` as a reference. This is computed as
  // a_t1_T_base_t1 * base_t2_T_a_t2. Here x_t indicates the pose of a frame x at time t.
  // Will return std::nullopt if `a` and `base` are not connected at either time t1 or time t2.
  std::optional<Pose3d> get(const Uuid& a, double t1, double t2, const Uuid& base) const {
    auto a1_T_base = get(a, base, t1);
    auto base_T_a2 = get(base, a, t2);
    if (!a1_T_base || !base_T_a2) {
      return std::nullopt;
    }
    return (*a1_T_base)*(*base_T_a2);
  }

  // The number of edges in the pose graph
  size_t numEdges() const {
    return histories_.size();
  }

  // The total number of poses stored in the pose graph
  size_t numEntries() const {
    size_t count = 0;
    for (const auto& history : histories_) {
      count += history.second.size();
    }
    return count;
  }

  // Gets a copy of the pose tree where every edge only contains the latest pose
  PoseTree latest() const;

  // Get direct access to all histories stored in the pose tree
  const std::map<std::pair<Uuid, Uuid>, PoseHistory>& histories() const { return histories_; }

 private:
  // Breadth-first search to find a path from lhs to rhs at the given timestamp. If it finds a path
  // it will compute and return the pose lhs_T_rhs, otherwise it will return std::nullopt.
  std::optional<Pose3d> findPath(const Uuid& lhs, const Uuid& rhs, double stamp) const;

  std::map<std::pair<Uuid, Uuid>, PoseHistory> histories_;
  std::map<Uuid, std::set<Uuid>> outgoing_;
};

}  // namespace pose_tree
}  // namespace isaac
