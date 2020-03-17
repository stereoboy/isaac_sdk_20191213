/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <memory>
#include <string>

#include "engine/alice/alice_codelet.hpp"
#include "engine/core/math/types.hpp"
#include "librealsense2/rs.hpp"
#include "messages/messages.hpp"

namespace isaac {

// RealsenseCamera is an Isaac codelet for the Realsense D435 camera that provides color and
// depth images. The sensor can also provide raw IR images, however this is currently not supported.
//
// You can change the resolution of the camera via various configuration parameters. However only
// certain modes are supported:
//  - 1280x720 (at most 30 Hz)
//  - 848x480
//  - 640x480
//  - 640x360
//  - 424x240
//
// Valid framerates for the color image are 60, 30, 15, 6 fps. Valid framerate for the depth image
// are 90, 60, 30, 15, 6 fps. The camera can also produce images at a 1080p resolution. However this
// is currently not supported as color and depth are set to the same resolution.
class RealsenseCamera : public alice::Codelet {
 public:
  RealsenseCamera();
  ~RealsenseCamera();

  void start() override;
  void tick() override;
  void stop() override;

  // A color camera image that can be Image3ub(for color) or Image1ui16 (for grayscale.)
  ISAAC_PROTO_TX(ColorCameraProto, color);
  // Depth image (in meters). This is in left Ir camera frame
  ISAAC_PROTO_TX(DepthCameraProto, depth);

  // The vertical resolution for both color and depth image.
  ISAAC_PARAM(int, rows, 360);
  // The horizontal resolution for both color and depth image.
  ISAAC_PARAM(int, cols, 640);
  // The framerate of color image acquisition.
  ISAAC_PARAM(int, rgb_framerate, 30);
  // The framerate of depth image acquisition.
  ISAAC_PARAM(int, depth_framerate, 30);
  // If enabled, the depth image is spatially aligned to the color image to provide matching color
  // and depth values for every pixel. This is a CPU-intensive process and can reduce frame rates.
  ISAAC_PARAM(bool, align_to_color, true);
  // Max number of frames you can hold at a given time. Increasing this number reduces frame
  // drops but increase latency, and vice versa; ranges from 0 to 32.
  ISAAC_PARAM(int, frame_queue_size, 2);
  // Limit exposure time when auto-exposure is ON to preserve constant fps rate.
  ISAAC_PARAM(bool, auto_exposure_priority, false);
  // Amount of power used by the depth laser, in mW. Valid ranges are between 0 and 360, in
  // increments of 30.
  ISAAC_PARAM(int, laser_power, 150);
  // Enable auto exposure, disabling can reduce motion blur
  ISAAC_PARAM(bool, enable_auto_exposure, true);
  // The index of the Realsense device in the list of devices detected. This indexing is dependent
  // on the order the Realsense library detects the cameras, and may vary based on mounting order.
  // By default the first camera device in the list is chosen. This camera choice can be overridden
  // by the serial number parameter below.
  ISAAC_PARAM(int, dev_index, 0)
  // An alternative way to specify the desired device in a multicamera setup. The serial number of
  // the Realsense camera can be found printed on the device. If specified, this parameter will take
  // precedence over the dev_index paramter above.
  ISAAC_PARAM(std::string, serial_number, "")

 private:
  // Inital configuration of a realsense device
  void initializeDeviceConfig(const rs2::device& dev);

  // Update configuration of a realsense device
  void updateDeviceConfig(const rs2::device& dev);

  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::RealsenseCamera);
