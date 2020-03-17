/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

namespace isaac {
namespace alice {

class Application;
class ApplicationJsonLoader;

// Loads websight and the sight API wrapper so that visualization via sight::show and Codelet::show
// are visible in the WebSight frontend.
void LoadWebSight(ApplicationJsonLoader& loader);

// Needs to be called in the main thread after the application was created. This is necessary to
// correctly initialize the sight::show API.
void InitializeSightApi(Application& app);

}  // namespace alice
}  // namespace isaac
