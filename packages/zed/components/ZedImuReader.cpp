/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "ZedImuReader.hpp"

#include "messages/math.hpp"
#include "packages/zed/ZedCamera.hpp"

namespace isaac {
namespace zed {

Pose3f GetCameraFromImuTransform(const sl::Transform& imu_T_camera, float scaling_factor) {
  const sl::Orientation orientation = imu_T_camera.getRotation();
  const sl::Translation translation = imu_T_camera.getTranslation() * scaling_factor;
  // ZED API uses floats
  const auto rotation = Quaternionf(orientation.w, orientation.x, orientation.y, orientation.z);
  return {SO3f::FromQuaternion(rotation), Vector3f(translation.x, translation.y, translation.z)};
}

void ZedImuReader::start() {
  camera_holder_ = node()->getComponentOrNull<ZedCamera>();
  if (camera_holder_ == nullptr) {
    reportFailure("[ZedImuReader] Can't obtain the Zed camera component.");
    return;
  }
  if (!camera_holder_->get_capture_imu_data()) {
    reportFailure("[ZedImuReader] Zed IMU data capture is disabled.");
    return;
  }
  // ZED Mini IMU poll rate is equal to the codelet tick frequency
  tickPeriodically();
}

void ZedImuReader::stop() {
  camera_holder_ = nullptr;
}

void ZedImuReader::tick() {
  auto camera = camera_holder_->getZedCamera();
  // ZED camera initialization process is performed in parallel to the ZedImuReader execution
  if (camera == nullptr) {
    return;
  }
  // We can't obtain camera information at ZedImuReader::start.
  const bool zed_mini = sl::MODEL_ZED_M == camera_holder_->getCameraInformation().camera_model;
  const bool capture_imu_data = camera_holder_->get_capture_imu_data();
  if (zed_mini && capture_imu_data) {
    sl::IMUData imu_data = {};
    sl::ERROR_CODE result = camera->getIMUData(imu_data, sl::TIME_REFERENCE_CURRENT);
    if (sl::SUCCESS == result) {
      publishImuData(imu_data);
    } else {
    // ZED API doesn't guarantee 100% reliability of every call.
    // Skipping a few IMU readings is not a critical problem for consumers
    // like visual inertial odometry
      LOG_WARNING("[ZedImuReader] Error capturing IMU data: %s", sl::toString(result).c_str());
    }

    // There's no need to publish camera_T_imu transform on every tick
    const int every_n_tick = get_imu_T_camera_publication_rate();
    if (getTickCount() % every_n_tick == 0) {
      auto msg = tx_imu_T_left_camera().initProto();
      const auto imu_T_camera = camera_holder_->getCameraInformation().camera_imu_transform;
      double scaling_factor = get_imu_translation_scaling_factor();
      ToProto(GetCameraFromImuTransform(imu_T_camera, scaling_factor).cast<double>(), msg);
      tx_imu_T_left_camera().publish();
    }
  }
}

void ZedImuReader::publishImuData(const sl::IMUData& imu_data) {
  auto imu_datamsg = tx_imu_raw().initProto();
  // set accelerometer data
  imu_datamsg.setLinearAccelerationX(imu_data.linear_acceleration.x);
  imu_datamsg.setLinearAccelerationY(imu_data.linear_acceleration.y);
  imu_datamsg.setLinearAccelerationZ(imu_data.linear_acceleration.z);

  // set gyroscope data
  imu_datamsg.setAngularVelocityX(DegToRad(imu_data.angular_velocity.x));
  imu_datamsg.setAngularVelocityY(DegToRad(imu_data.angular_velocity.y));
  imu_datamsg.setAngularVelocityZ(DegToRad(imu_data.angular_velocity.z));

  const int64_t isaac_time =
      camera_holder_->zedToIsaacTimestamp(imu_data.timestamp);
  tx_imu_raw().publish(isaac_time);

  // Show IMU data in Sight
  show("RawLinearAcceleration.x", imu_data.linear_acceleration.x);
  show("RawLinearAcceleration.y", imu_data.linear_acceleration.y);
  show("RawLinearAcceleration.z", imu_data.linear_acceleration.z);
  show("RawAngularVelocity.x", imu_data.angular_velocity.x);
  show("RawAngularVelocity.y", imu_data.angular_velocity.y);
  show("RawAngularVelocity.z", imu_data.angular_velocity.z);
}

}  // namespace zed
}  // namespace isaac
