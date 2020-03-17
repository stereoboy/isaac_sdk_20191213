/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(Alice, AppLoadEmpty) {
  const char app_json_text[] =
R"???({
  "name": "app_load_test"
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));
  EXPECT_EQ("app_load_test", app.name());
  app.startWaitStop(.1);
}


TEST(Alice, AppLoadEmptyWithScheduler) {
  const char app_json_text[] =
R"???({
  "name": "app_load_test",
  "scheduler": {
    "execution_groups": [
      {
        "name": "MyTestWorkerGroup",
        "cores": [0,1,2,3],
        "workers": true
      },
      {
        "name": "MyTestBlockerGroup",
        "cores": [4,5,6,7],
        "workers": false
      }
    ]
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));
  EXPECT_EQ("app_load_test", app.name());
  app.startWaitStop(.1);
}

TEST(Alice, AppLoadEmptyWithSchedulerTime) {
  const char app_json_text[] =
R"???({
  "name": "app_load_test",
  "scheduler": {
    "use_time_machine": true,
    "clock_scale": 1.0,
    "execution_groups": [
      {
        "name": "MyTestWorkerGroup",
        "cores": [0,1,2,3],
        "workers": true
      },
      {
        "name": "MyTestBlockerGroup",
        "cores": [4,5,6,7],
        "workers": false
      }
    ]
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));
  EXPECT_EQ("app_load_test", app.name());
  app.startWaitStop(.1);
}


TEST(Alice, AppLoadEmptyWithDefaultExecutionGroup) {
  const char app_json_text[] =
R"???({
  "name": "app_load_test",
  "scheduler": {
    "use_time_machine": true,
    "clock_scale": 1.0,
    "default_execution_group_config": [
      {
        "worker_cores": [0,1],
        "blocker_cores": [4,5]
      }
    ]
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));
  EXPECT_EQ("app_load_test", app.name());
  app.startWaitStop(.1);
}



}  // namespace alice
}  // namespace isaac
