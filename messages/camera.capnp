#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0xf989d1556d157216;

using import "math.capnp".Vector2dProto;
using import "math.capnp".Vector5fProto;
using import "image.capnp".ImageProto;

# An image published by a color (or greyscale) camera
struct ColorCameraProto {
  # Actual image captured by the camera. The pixel type depends on the type of camera. Most commonly
  # it could be a single 8-bit integer for a greyscale camera, or a three 8-bit integers for a
  # color camera.
  image @0: ImageProto;

  # The choices available for color space
  enum ColorSpace {
    greyscale @0;
    rgb @1;
    bgr @2;
    yuv @3;
    rgba @4;
  }

  # Color space used by the image
  colorSpace @1: ColorSpace;

  # Intrinsic camera parameters
  pinhole @2: PinholeProto;
  distortion @3: DistortionProto;
}

# Pinhole camera model parameters
struct PinholeProto {
  # Resolution of the camera
  rows @0: Int16;
  cols @1: Int16;

  # Focal length of the camera in pixel (0: row, 1: col)
  focal @2: Vector2dProto;

  # Optical center of the camera in pixel (0: row, 1: col)
  center @3: Vector2dProto;
}

# Distortion parameters for a camera image
struct DistortionProto {
  # The choices available for the distortion model
  enum DistortionModel {
    # Brown distortion model with 3 radial and 2 tangential distortion
    # coefficients: (r0 r1 r2 t0 t1)
    brown   @0;
    # Fisheye (wide-angle) distortion. 4 radial (r0, r1, r2, r3) distortion coefficients
    fisheye @1;
  }

  # Distortion model and coefficients
  model @0: DistortionModel;
  coefficients @1: Vector5fProto;
}

# Depth information for an image
struct DepthCameraProto {
  # A depth value for every pixel in the image. The pixel type is a single 32-bit float.
  depthImage @0: ImageProto;
  # The minimum and maximum depth used
  minDepth @1: Float32;
  maxDepth @2: Float32;
  # Intrinsic camera parameters
  pinhole @3: PinholeProto;
}

# Pixel-wise class label and instance segmentations for an image
struct SegmentationCameraProto {
  # A image which stores a class label for every pixel. The pixel type is a single unsigned 8-bit
  # integer.
  labelImage @0: ImageProto;
  # A label used to enumerate all labels in the `labels` field below
  struct Label {
    # The integer value stored in the `labelImage`
    index @0: UInt8;
    # The name of the label
    name @1: Text;
  }
  # List of all labels used in labelImage
  labels @1: List(Label);
  # A image which stores an instance index for every pixel. Different objects with the same
  # label will get different instance indices. The pixel type is a single unsigned 16-bit integer.
  instanceImage @2: ImageProto;
  # Intrinsic camera parameters
  pinhole @3: PinholeProto;
}

# Disparity information for an image
struct DisparityCameraProto {
  # A disparity value for every pixel in the image. The pixel type is a single 32-bit float.
  disparityImage @0: ImageProto;

  # The minimum value that a pixel could be in the disparityImage.
  # This should be greater than zero so that a maximum depth can be computed.
  minimumDisparity @1: Float32;

  # The maximum value that a pixel could be in the disparityImage.
  maximumDisparity @2: Float32;

  # The distance between the two stereo cameras (in meters).
  cameraBaseline @3: Float32;

  # Intrinsic camera parameters
  pinhole @4: PinholeProto;
}
