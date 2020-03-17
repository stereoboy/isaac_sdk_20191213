/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "pose_tree.hpp"

#include <map>
#include <queue>
#include <utility>

#include "engine/core/assert.hpp"
#include "engine/core/math/pose3.hpp"
#include "engine/gems/interpolation/poses.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace pose_tree {

namespace {

constexpr const size_t kHistoryMaxSize = 1000;

// Interpolates a pose in a time series at given time.
std::optional<Pose3d> Interpolate(const PoseTree::PoseHistory& history, double stamp) {
  return history.interpolate_2_p(stamp,
      [](double p, const Pose3d& a, const Pose3d& b) {
        return ::isaac::Interpolate(p, a, b);
      });
}
Pose3d Interpolate(const PoseTree::PoseHistory& history, double stamp, size_t index) {
  return history.interpolate_2_p(stamp, index,
      [](double p, const Pose3d& a, const Pose3d& b) {
        return ::isaac::Interpolate(p, a, b);
      });
}

}  // namespace

bool PoseTree::hasDirectConnection(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  if (lhs == rhs) {
    return true;
  }
  return histories_.find({lhs, rhs}) != histories_.end();
}

bool PoseTree::hasIndirectConnection(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  return !hasDirectConnection(lhs, rhs, stamp)
      && findPath(lhs, rhs, stamp);
}

std::optional<Pose3d> PoseTree::get(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  // Check for identity
  if (lhs == rhs) {
    return Pose3d::Identity();
  }
  // See if this is a direct edge
  auto it = histories_.find({lhs, rhs});
  if (it != histories_.end()) {
    return Interpolate(it->second, stamp);
  }
  // try to find a path
  return findPath(lhs, rhs, stamp);
}

bool PoseTree::set(const Uuid& lhs, const Uuid& rhs, double stamp, const Pose3d& lhs_T_rhs) {
  if (lhs == rhs) {
    return false;
  }
  if (hasIndirectConnection(lhs, rhs, stamp)) {
    return false;
  }
  auto& h_lhs_rhs = histories_[{lhs, rhs}];
  auto& h_rhs_lhs = histories_[{rhs, lhs}];
  h_lhs_rhs.insert(stamp, lhs_T_rhs);
  h_rhs_lhs.insert(stamp, lhs_T_rhs.inverse());
  h_lhs_rhs.forgetBySize(kHistoryMaxSize);
  h_rhs_lhs.forgetBySize(kHistoryMaxSize);
  outgoing_[lhs].insert(rhs);
  outgoing_[rhs].insert(lhs);
  return true;
}

PoseTree PoseTree::latest() const {
  PoseTree result;
  result.outgoing_ = outgoing_;
  for (const auto& kvp : histories_) {
    if (kvp.second.empty()) continue;
    const auto& latest = kvp.second.youngest();
    result.histories_[kvp.first].push(latest.stamp, latest.state);
  }
  return result;
}

std::optional<Pose3d> PoseTree::findPath(const Uuid& lhs, const Uuid& rhs, double stamp) const {
  if (outgoing_.count(lhs) == 0 || outgoing_.count(rhs) == 0) {
    return std::nullopt;
  }
  struct BreadCrumb {
    Uuid parent;
    ssize_t history_index;
  };
  std::queue<Uuid> open;
  open.push(lhs);
  std::map<Uuid, BreadCrumb> visited;
  visited[lhs] = {lhs, -1};
  while (!open.empty()) {
    Uuid top = std::move(open.front());
    open.pop();
    auto it = outgoing_.find(top);
    ASSERT(it != outgoing_.end(), "Node without outgoing edges");
    for (const Uuid& out : it->second) {
      // Check if already visited
      if (visited.count(out) > 0) {
        continue;
      }
      // New possible edge
      auto jt = histories_.find({top, out});
      ASSERT(jt != histories_.end(), "Edge without history");
      const ssize_t history_lower_index = jt->second.interpolate_2_index(stamp);
      if (history_lower_index == -1) {
        continue;
      }
      // Check if target found
      if (out == rhs) {
        // trace back path
        Uuid back = top;
        Pose3d back_T_rhs = Interpolate(jt->second, stamp, history_lower_index);
        while (back != lhs) {
          auto kt = visited.find(back);
          ASSERT(kt != visited.end(), "How did we come here?");
          jt = histories_.find({kt->second.parent, back});
          ASSERT(jt != histories_.end(), "Edge without history");
          const Pose3d prev_T_back = Interpolate(jt->second, stamp, kt->second.history_index);
          back_T_rhs = prev_T_back * back_T_rhs;
          back = kt->second.parent;
        }
        return back_T_rhs;
      }
      // Add possible branch
      open.push(out);
      visited[out] = {top, history_lower_index};
    }
  }
  return std::nullopt;
}

}  // namespace pose_tree
}  // namespace isaac
