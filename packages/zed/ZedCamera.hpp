/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

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
#include "engine/core/image/image.hpp"
#include "engine/gems/geometry/pinhole.hpp"
#include "messages/messages.hpp"
#include "sl/Camera.hpp"

namespace sl {

NLOHMANN_JSON_SERIALIZE_ENUM(RESOLUTION, {
    {RESOLUTION_HD2K, "2208x1242"},
    {RESOLUTION_HD1080, "1920x1080"},
    {RESOLUTION_HD720, "1280x720"},
    {RESOLUTION_VGA, "672x376"},
    {RESOLUTION_LAST, nullptr},
});

}  // namespace sl

namespace isaac {

// Provides stereo image pairs and calibration information from a ZED camera
class ZedCamera : public alice::Codelet {
 public:
  void start() override;
  void tick() override;
  void stop() override;

  // Returns the ZED camera object which serves as an entry point to ZED SDK API calls
  sl::Camera* getZedCamera() { return zed_.get(); }

  // Returns the ZED camera information
  const sl::CameraInformation& getCameraInformation() const { return zed_info_; }

  // Transforms ZED clock reading into ISAAC clock reading
  int64_t zedToIsaacTimestamp(int64_t zed_timestamp) const {
      return timestamp_offset_ + zed_timestamp;
  }

  // left rgb image and camera intrinsics
  ISAAC_PROTO_TX(ColorCameraProto, left_camera_rgb);
  // right rgb image and camera intrinsics
  ISAAC_PROTO_TX(ColorCameraProto, right_camera_rgb);
  // left gray image and camera intrinsics
  ISAAC_PROTO_TX(ColorCameraProto, left_camera_gray);
  // right gray rgb image and camera intrinsics
  ISAAC_PROTO_TX(ColorCameraProto, right_camera_gray);
  // camera pair extrinsics (right-to-left)
  ISAAC_PROTO_TX(Pose3dProto, extrinsics);

  // The resolution to use for the ZED camera. The following values can be set:
  //  RESOLUTION_HD2K:   2208x1242
  //  RESOLUTION_HD1080: 1920x1080
  //  RESOLUTION_HD720:  1280x720
  //  RESOLUTION_VGA:    672x376
  ISAAC_PARAM(sl::RESOLUTION, resolution, sl::RESOLUTION_VGA);
  // The image frame rate for the ZED camera.
  // If set to 0, the highest FPS of the specified camera_resolution will be used.
  // The list of supported resolution - framerate combinations:
  // RESOLUTION_HD2K, /**< 2208*1242, available framerates: 15 fps.*/
  // RESOLUTION_HD1080, /**< 1920*1080, available framerates: 15, 30 fps.*/
  // RESOLUTION_HD720, /**< 1280*720, available framerates: 15, 30, 60 fps.*/
  // RESOLUTION_VGA, /**< 672*376, available framerates: 15, 30, 60, 100 fps.*/
  // If the requested camera_fps is unsuported, the closest available FPS will be used.
  // ZED Camera FPS is not tied to a codelet tick rate as the camera has an independent on-board CPU
  ISAAC_PARAM(int, camera_fps, 60);
  // The numeral of the system video device of the ZED camera. For example for /dev/video0 choose 0.
  ISAAC_PARAM(int, device_id, 0);
  // Turns on capture and publication of IMU data that is only supported by ZED Mini camera hardware
  ISAAC_PARAM(bool, capture_imu_data, false);
  // The folder path to the settings file (SN#####.conf) for the zed camera.
  // This file contains the calibration parameters for the camera.
  ISAAC_PARAM(std::string, settings_folder_path, "./");
  // The GPU device to be used for ZED CUDA operations
  ISAAC_PARAM(int, gpu_id, 0);
  // Turns on gray scale images
  ISAAC_PARAM(bool, gray_scale, false);
  // Turns on RGB color images
  ISAAC_PARAM(bool, rgb, true);
  // Turns on rectification of images inside ZED camera
  ISAAC_PARAM(bool, enable_factory_rectification, true);

 private:
  // Initializes the sl::Camera object which serves as an entry point to ZED SDK API calls.
  // This method is called once at startup.
  // The codelet execution is fail-stopped if the ZED camera can't be initialized.
  bool initializeZedCamera();
  // Copy image data into the class fields
  void retriveImages();
  // Publish the RGB or monochrome stereo image data
  void publishImageData(int64_t acq_time);
  // Publish the monochrome stereo image data
  void publishGrayData(int64_t acq_time, const sl::CalibrationParameters& camera_params);
  // Publish the RGB stereo image data
  void publishRgbData(int64_t acq_time, const sl::CalibrationParameters& camera_params);
  // Publish ZED camera intrinsics and extrinsics
  void publishCameraData(int64_t acq_time);

  // ZED camera state
  std::unique_ptr<sl::Camera> zed_;
  sl::CameraInformation zed_info_ = {};
  sl::Mat left_image_rgb_;
  sl::Mat right_image_rgb_;
  sl::Mat left_image_gray_;
  sl::Mat right_image_gray_;
  sl::Resolution zed_resolution_ = {};
  sl::RuntimeParameters zed_run_params_ = {};
  int64_t timestamp_offset_;
};

}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::ZedCamera);
