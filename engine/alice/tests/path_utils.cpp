/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/utils/path_utils.hpp"
#include "engine/alice/alice.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

TEST(path_utils, TranslateAssetPath_isaac) {
  const std::vector<std::string> asset_inputs{
      "foo",                              //
      "packages/foo",                     //
      "@workspace//packages/foo",         //
      "@com_nvidia_isaac//packages/foo",  //
  };
  // Output building in isaac workspace
  const std::vector<std::string> asset_outputs{
      "foo",                              //
      "packages/foo",                     //
      "external/workspace/packages/foo",  //
      "packages/foo",                     //
  };
  const std::vector<std::string> asset_workspace_outputs{
      "com_nvidia_isaac",  //
      "com_nvidia_isaac",  //
      "workspace",         //
      "com_nvidia_isaac",  //
  };

  const std::string cur_workspace = "com_nvidia_isaac";
  const std::string home_workspace = "com_nvidia_isaac";

  int idx = 0;
  for (const auto& in : asset_inputs) {
    const std::string workspace_out = TranslateAssetPath(in, home_workspace, cur_workspace).first;
    const std::string asset_out = TranslateAssetPath(in, home_workspace, cur_workspace).second;
    EXPECT_EQ(asset_outputs[idx], asset_out);
    EXPECT_EQ(asset_workspace_outputs[idx], workspace_out);
    idx++;
  }
}

TEST(path_utils, TranslateAssetPath_external) {
  const std::vector<std::string> asset_inputs{
      "foo",                              //
      "packages/foo",                     //
      "@workspace//packages/foo",         //
      "@com_nvidia_isaac//packages/foo",  //
  };
  // Output building in external workspace
  const std::vector<std::string> external_asset_outputs{
      "external/com_nvidia_isaac/foo",           //
      "external/com_nvidia_isaac/packages/foo",  //
      "packages/foo",                            //
      "external/com_nvidia_isaac/packages/foo",  //
  };
  const std::vector<std::string> external_asset_workspace_outputs{
      "com_nvidia_isaac",  //
      "com_nvidia_isaac",  //
      "workspace",         //
      "com_nvidia_isaac",  //
  };

  const std::string external_cur_workspace = "com_nvidia_isaac";
  const std::string external_home_workspace = "workspace";

  int idx = 0;
  for (const auto& in : asset_inputs) {
    const std::string workspace_out =
        TranslateAssetPath(in, external_home_workspace, external_cur_workspace).first;
    const std::string asset_out =
        TranslateAssetPath(in, external_home_workspace, external_cur_workspace).second;
    EXPECT_EQ(external_asset_outputs[idx], asset_out);
    EXPECT_EQ(external_asset_workspace_outputs[idx], workspace_out);
    idx++;
  }
}

TEST(path_utils, ExpandModulePath) {
  int idx = 0;
  const std::vector<std::string> module_inputs{
      "foo",                             //
      "packages/foo",                    //
      "packages/foo:bar",                //
      "@workspace//packages/foo",        //
      "@com_nvidia_isaac//packages/foo"  //
  };
  const std::vector<std::string> module_outputs{
      "packages/foo/libfoo_module.so",                     //
      "packages/foo/libfoo_module.so",                     //
      "packages/foo/libbar_module.so",                     //
      "external/workspace/packages/foo/libfoo_module.so",  //
      "packages/foo/libfoo_module.so"                      //
  };
  const std::vector<std::string> external_module_outputs{
      "external/com_nvidia_isaac/packages/foo/libfoo_module.so",  //
      "external/com_nvidia_isaac/packages/foo/libfoo_module.so",  //
      "external/com_nvidia_isaac/packages/foo/libbar_module.so",  //
      "packages/foo/libfoo_module.so",                            //
      "external/com_nvidia_isaac/packages/foo/libfoo_module.so"   //
  };
  const std::vector<std::string> default_module_outputs{
      "packages/foo/libfoo_module.so",                           //
      "packages/foo/libfoo_module.so",                           //
      "packages/foo/libbar_module.so",                           //
      "external/workspace/packages/foo/libfoo_module.so",        //
      "external/com_nvidia_isaac/packages/foo/libfoo_module.so"  //
  };

  const std::string cur_workspace = "com_nvidia_isaac";
  const std::string home_workspace = "com_nvidia_isaac";

  idx = 0;
  for (const auto& in : module_inputs) {
    const std::string module_out = ExpandModulePath(in, home_workspace, cur_workspace);
    EXPECT_EQ(module_outputs[idx], module_out);
    idx++;
  }

  const std::string external_cur_workspace = "com_nvidia_isaac";
  const std::string external_home_workspace = "workspace";
  idx = 0;
  for (const auto& in : module_inputs) {
    const std::string module_out =
        ExpandModulePath(in, external_home_workspace, external_cur_workspace);
    EXPECT_EQ(external_module_outputs[idx], module_out);
    idx++;
  }

  idx = 0;
  for (const auto& in : module_inputs) {
    const std::string module_out = ExpandModulePath(in);
    EXPECT_EQ(default_module_outputs[idx], module_out);
    idx++;
  }
}

}  // namespace alice
}  // namespace isaac
