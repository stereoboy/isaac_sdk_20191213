/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "websight.hpp"

#include <string>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/application_json_loader.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/modules.hpp"
#include "engine/gems/sight/sight.hpp"
#include "packages/sight/AliceSight.hpp"

namespace isaac {
namespace alice {

namespace {

// The following string defines the websight graph and basic configuration
constexpr char kWebsightJsonText[] = R"???(
{
  "modules": [
    "sight"
  ],
  "config": {
    "_statistics": {
      "NodeStatistics": {
        "tick_period": "1 Hz"
      }
    },
    "_pose_tree_bridge": {
      "PoseTreeJsonBridge": {
        "tick_period": "50ms"
      }
    },
    "_interactive_markers_bridge": {
      "InteractiveMarkersBridge": {
        "tick_period": "50ms"
      }
    }
  },
  "graph": {
    "nodes": [
      {
        "name": "websight",
        "start_order": -1100,
        "components": [
          {
            "name": "isaac.alice.MessageLedger",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "WebsightServer",
            "type": "isaac::sight::WebsightServer"
          },
          {
            "name": "isaac.alice.SightChannelStatus",
            "type": "isaac::alice::SightChannelStatus"
          },
          {
            "name": "isaac.sight.AliceSight",
            "type": "isaac::sight::AliceSight"
          }
        ]
      },
      {
        "name": "_config_bridge",
        "start_order": -1000,
        "components": [
          {
            "name": "isaac.alice.MessageLedger",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "isaac.alice.ConfigBridge",
            "type": "isaac::alice::ConfigBridge"
          }
        ]
      },
      {
        "name": "_statistics",
        "start_order": -1000,
        "components": [
          {
            "name": "isaac.alice.MessageLedger",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "NodeStatistics",
            "type": "isaac::alice::NodeStatistics"
          }
        ]
      },
      {
        "name": "_pose_tree_bridge",
        "start_order": -1000,
        "components": [
          {
            "name": "isaac.alice.MessageLedger",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "PoseTreeJsonBridge",
            "type": "isaac::alice::PoseTreeJsonBridge"
          }
        ]
      },
      {
        "name": "_interactive_markers_bridge",
        "start_order": -1000,
        "components": [
          {
            "name": "isaac.alice.MessageLedger",
            "type": "isaac::alice::MessageLedger"
          },
          {
            "name": "InteractiveMarkersBridge",
            "type": "isaac::alice::InteractiveMarkersBridge"
          }
        ]
      }
    ],
    "edges": [
      {
        "source": "websight/WebsightServer/config",
        "target": "_config_bridge/isaac.alice.ConfigBridge/request"
      },
      {
        "source": "_config_bridge/isaac.alice.ConfigBridge/reply",
        "target": "websight/WebsightServer/config_reply"
      },
      {
        "source": "websight/WebsightServer/statistics",
        "target": "_statistics/NodeStatistics/request"
      },
      {
        "source": "_statistics/NodeStatistics/statistics",
        "target": "websight/WebsightServer/statistics_reply"
      },
      {
        "source": "_pose_tree_bridge/PoseTreeJsonBridge/pose_tree",
        "target": "websight/WebsightServer/pose_tree_reply"
      },
      {
        "source": "websight/WebsightServer/interactive_markers",
        "target": "_interactive_markers_bridge/InteractiveMarkersBridge/request"
      },
      {
        "source": "_interactive_markers_bridge/InteractiveMarkersBridge/reply",
        "target": "websight/WebsightServer/interactive_markers_reply"
      }
    ]
  }
})???";

}  // namespace

void LoadWebSight(ApplicationJsonLoader& loader) {
  LOG_INFO("Loading websight...");
  const auto json = serialization::ParseJson(kWebsightJsonText);
  ASSERT(json, "Error in Websight JSON text");
  loader.loadMore(*json);
}

void InitializeSightApi(Application& app) {
  // Set the correct pointer for the raw sight interface
  sight::ResetSight(app.findComponentByName<sight::AliceSight>("websight/isaac.sight.AliceSight"));
}

}  // namespace alice
}  // namespace isaac
