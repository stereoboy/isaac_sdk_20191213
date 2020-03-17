/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/

#include <string>
#include <vector>

#include "engine/alice/alice.hpp"
#include "engine/alice/tools/parse_command_line.hpp"
#include "engine/alice/tools/websight.hpp"
#include "engine/core/time.hpp"
#include "gflags/gflags.h"

DEFINE_string(duration, "", "Duration to run the app for. <number>[s|m|h]");

// A default main file for an Isaac application. This can be used to execute an application defined
// in a JSON file.
int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  ::isaac::alice::Application app(isaac::alice::ParseApplicationCommandLine());
  ::isaac::alice::InitializeSightApi(app);

  std::optional<double> duration = isaac::ParseDurationStringToSecond(FLAGS_duration);

  if (!duration) {
    app.startWaitStop();
  } else {
    app.startWaitStop(*duration);
  }
  return 0;
}
