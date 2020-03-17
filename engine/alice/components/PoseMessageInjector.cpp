/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/components/PoseMessageInjector.hpp"

#include <string>

#include "messages/math.hpp"

namespace isaac {
namespace alice {

void PoseMessageInjector::start() {
  tickOnMessage(rx_pose());
}

void PoseMessageInjector::tick() {
  rx_pose().processAllNewMessages([&](auto proto, int64_t pubtime, int64_t acqtime) {
    const std::string lhs = proto.getLhs();
    const std::string rhs = proto.getRhs();
    const bool ok = node()->pose().set(lhs, rhs, FromProto(proto.getPose()), ToSeconds(acqtime));
    if (!ok) {
      LOG_ERROR("Could not set pose %s_T_%s. Maybe another path in the tree already exists.",
                lhs.c_str(), rhs.c_str());
    }
  });
}

}  // namespace alice
}  // namespace isaac
