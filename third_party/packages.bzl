"""
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
"""

load("//engine/build:isaac.bzl", "isaac_git_repository", "isaac_new_git_repository")
load("//engine/build:isaac.bzl", "isaac_http_archive", "isaac_new_http_archive")
load("//engine/build:isaac.bzl", "isaac_new_local_repository")

def clean_dep(dep):
    return str(Label(dep))

# loads dependencies for various modules
def isaac_packages_workspace():
    isaac_new_http_archive(
        name = "robotis",
        build_file = clean_dep("//third_party:dynamixel.BUILD"),
        sha256 = "1233525218b59ee9b923124ca688feab7014362c1c9c7ad4a844927f8ec3dba5",
        url = "https://developer.nvidia.com/isaac/download/third_party/robotis_dynamixel_sdk-3-6-2-tar-gz",
        type = "tar.gz",
        strip_prefix = "DynamixelSDK-3.6.2",
        licenses = ["@robotis//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "assimp",
        build_file = clean_dep("//third_party:assimp.BUILD"),
        sha256 = "60080d8ab4daaab309f65b3cffd99f19eb1af8d05623fff469b9b652818e286e",
        url = "https://developer.nvidia.com/isaac/download/third_party/assimp-4-0-1-tar-gz",
        type = "tar.gz",
        strip_prefix = "assimp-4.0.1",
        licenses = ["@assimp//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "nvstereomatcher",
        build_file = clean_dep("//third_party:nvstereomatcher.BUILD"),
        sha256 = "c4db7e3641d32f370394570181c5f85fc33667a195c5de7f6bef8d4194e315af",
        url = "https://developer.nvidia.com/isaac/download/third_party/libnvstereomatcher_v5-tar-gz",
        type = "tar.gz",
        strip_prefix = "libnvstereomatcher_v5",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "apriltags",
        build_file = clean_dep("//third_party:apriltags.BUILD"),
        sha256 = "5cfc4bc73fccc4d737685e345060e98e2ce067ac41026f4a1e212b402e6c33c0",
        url = "https://developer.nvidia.com/isaac/download/third_party/april_tag_v5-tar-gz",
        type = "tar.gz",
        strip_prefix = "libapriltagging_v5",
        licenses = ["//:LICENSE"],
    )

    isaac_git_repository(
        name = "gmapping_repo",
        remote = "https://github.com/lullabee/openslam_gmapping.git",
        commit = "6f2ac5a2a2a8637ee844b4096f288f50d27a24cb",
        licenses = ["https://openslam-org.github.io/gmapping.html"],
    )

    native.bind(
        name = "gmapping",
        actual = "@gmapping_repo//:gmapping",
    )

    isaac_http_archive(
        name = "audio_assets",
        sha256 = "3915240ad6c5fe50f50a84204b6d9b602505f2558fad4c14b27187266f458b25",
        url = "https://developer.nvidia.com/isaac/download/third_party/alsa_audio_assets-v2-tar-gz",
        type = "tar.gz",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_http_archive(
        name = "torch_inference_test_assets",
        build_file = clean_dep("//third_party:torch_inference_test_assets.BUILD"),
        sha256 = "24e10fbb2aae938b9dcfbaa8ceb1fc65b31de33733159c23a1e1d3c545cb8322",
        url = "https://developer.nvidia.com/isaac/download/third_party/torch_inference_test_assets-v2-tar-gz",
        type = "tar.gz",
        licenses = ["//:LICENSE"],
    )

    isaac_http_archive(
        name = "ball_segmentation_model",
        build_file = clean_dep("//third_party:ball_segmentation_model.BUILD"),
        sha256 = "b0b01e06a0b02f316748d664c9b07b1dbd0ea70dc692262d9df2ac1fdbd22ddd",
        url = "https://developer.nvidia.com/isaac/download/third_party/ball_segmentation_toy_model-20190626-tar-xz",
        type = "tar.xz",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "voice_command_detection_model_carter",
        build_file = clean_dep("//third_party:voice_command_detection_model_carter.BUILD"),
        sha256 = "57e1b0f70136f7008b467d02eb97d8f09da45e85ca6a8cb442aca9ea2f3d7b55",
        url = "https://developer.nvidia.com/isaac/download/third_party/vcd_model_carter_v1-tar-gz",
        type = "tar.gz",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "voice_command_detection_model_kaya",
        build_file = clean_dep("//third_party:voice_command_detection_model_kaya.BUILD"),
        sha256 = "80a8251c81735c88573e17933f553da2aead04771fea2dee76348eddc85d426d",
        url = "https://developer.nvidia.com/isaac/download/third_party/vcd_model_kaya_v1-tar-gz",
        type = "tar.gz",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "resnet_object_detection_model",
        build_file = clean_dep("//third_party:resnet_object_detection_model.BUILD"),
        sha256 = "c15f5536062a755ffe8dd5ee7425c07f382849c2d5d0d5f1a6505d0904730473",
        url = "https://developer.nvidia.com/isaac/download/third_party/resnet18_detector_dolly_20191122-zip",
        type = "zip",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "dolly_pose_estimation_model",
        build_file = clean_dep("//third_party:dolly_pose_estimation_model.BUILD"),
        sha256 = "40cf8edf6e15dc6544b3c12a151d68391038f6bf18a163049803ecef59d3a660",
        url = "https://developer.nvidia.com/isaac/download/third_party/dolly_pose_estimation_model_v3-tar-gz",
        type = "tar.gz",
        strip_prefix = "dolly_pose_estimation_model_v2",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "dolly_detection_pose_estimation_data",
        build_file = clean_dep("//third_party:dolly_detection_pose_estimation_data.BUILD"),
        sha256 = "02c48f49ae364ec92e022a8bd03db1b0da034ca0fc9001cabff7294ba89c4582",
        url = "https://developer.nvidia.com/isaac/download/third_party/dolly_detection_pose_estimation_data_v3-tar-gz",
        type = "tar.gz",
        strip_prefix = "dolly_detection_pose_estimation_data_v3",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "openpose_model",
        build_file = clean_dep("//third_party:openpose_model.BUILD"),
        sha256 = "134ac4553d34edf61b5cb91c5db4c124f87ce7462de3a08823c7c4fbee21ce1d",
        url = "https://developer.nvidia.com/isaac/download/third_party/ix-networks-openpose-20190815-tar-xz",
        type = "tar.xz",
        licenses = ["https://creativecommons.org/publicdomain/zero/1.0/legalcode"],
    )

    isaac_new_http_archive(
        name = "openpose_trt_pose_model",
        build_file = clean_dep("//third_party:openpose_model.BUILD"),
        sha256 = "da2e33a16a5d792e0d2983ed26a78be86909d8673e0b9b2035bf5bc2841c59f6",
        url = "https://developer.nvidia.com/isaac/download/third_party/trt-pose-20191107-tar-xz",
        type = "tar.xz",
        licenses = ["https://github.com/NVIDIA-AI-IOT/trt_pose/blob/master/LICENSE.md"],
    )

    isaac_new_http_archive(
        name = "alsa",
        build_file = clean_dep("//third_party:alsa.BUILD"),
        sha256 = "938832b91e5ac8c4aee9847561f680814d199ba5ad9fb795c5a699075a19fd61",
        url = "https://developer.nvidia.com/isaac/download/third_party/alsa-x86_64-tar-xz",
        type = "tar.xz",
        licenses = ["https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html"],
    )

    # Used for both TX2 and Xavier
    isaac_new_http_archive(
        name = "alsa_aarch64",
        build_file = clean_dep("//third_party:alsa.BUILD"),
        sha256 = "8b0b1f65bc7fbdf45c30389457c530c423518dd12b32cdddca704bfd0daf0ec9",
        url = "https://developer.nvidia.com/isaac/download/third_party/alsa-aarch64-tar-xz",
        type = "tar.xz",
        licenses = ["https://www.gnu.org/licenses/old-licenses/lgpl-2.1.en.html"],
    )

    isaac_new_http_archive(
        name = "libi2c_aarch64",
        build_file = clean_dep("//third_party:libi2c.BUILD"),
        sha256 = "0371eb3a1f60a5515f5571e5fc07711eb5d82f575060096fae09dcd0821b7d39",
        url = "https://developer.nvidia.com/isaac/download/third_party/libi2c-0-aarch64_xavier-tar-xz",
        type = "tar.xz",
        strip_prefix = "libi2c",
        licenses = ["https://raw.githubusercontent.com/amaork/libi2c/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "vrworks_warp360",
        build_file = clean_dep("//third_party:warp360.BUILD"),
        sha256 = "48225cc6bae5a50f342998cd7bde5015f3402f7371d3c3c8deda23921171d532",
        url = "https://developer.nvidia.com/isaac/download/third_party/vrworks_warp360-3-tar-gz",
        type = "tar.gz",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "libargus",
        build_file = clean_dep("//third_party:libargus.BUILD"),
        sha256 = "8db8df094efb31a6c945e69001d6f9de3ccdbe8752d4069ef05c79c14ec0af5b",
        url = "https://developer.nvidia.com/isaac/download/third_party/libargus-2019-02-tar-gz",
        type = "tar.gz",
        strip_prefix = "libargus",
        licenses = ["https://raw.githubusercontent.com/pauldotknopf/JetsonTX1Drivers/master/nv_tegra/LICENSE.libargus"],
    )

    isaac_new_http_archive(
        name = "vicon_datastream",
        build_file = clean_dep("//third_party:vicon_datastream.BUILD"),
        sha256 = "f8e0d88ad53a99e3ef4de21891781c664fb333a7e656967fd1d4230d7538371e",
        url = "https://developer.nvidia.com/isaac/download/third_party/vicon-datastream-sdk-tar-gz",
        type = "tar.gz",
        licenses = ["https://www.vicon.com/products/software/datastream-sdk"],
    )

    isaac_new_http_archive(
        name = "elbrus_vo",
        build_file = clean_dep("//third_party:elbrus_vo.BUILD"),
        sha256 = "803fee4263f82d0bff7c37d141c3bdf8be1934fea393899d1bcb0be06f060438",
        url = "https://developer.nvidia.com/isaac/download/third_party/elbrus_v6_1-tar-xz",
        type = "tar.xz",
        strip_prefix = "elbrus",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "kinova_jaco",
        build_file = clean_dep("//third_party:kinova_jaco.BUILD"),
        sha256 = "a8fa1a09ec98a69ab508176c35e582ed41abb63da430c73b8371940c68739fdd",
        url = "https://developer.nvidia.com/isaac/download/third_party/kinova_ros-1-2-1-tar-gz",
        type = "tar.gz",
        strip_prefix = "kinova-ros-1.2.1",
        licenses = ["https://raw.githubusercontent.com/Kinovarobotics/kinova-ros/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "realsense",
        build_file = clean_dep("//third_party:realsense.BUILD"),
        sha256 = "5dafabd13fe3ed23ae6c1f6c7f0c902de580f3a60a8b646e9868f7edc962abf2",
        url = "https://developer.nvidia.com/isaac/download/third_party/librealsense-v2-29-0-tar-gz",
        type = "tar.gz",
        strip_prefix = "librealsense-2.29.0",
        licenses = ["@realsense//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "opencv_x86_64",
        build_file = clean_dep("//third_party:opencv.BUILD"),
        sha256 = "364b6004167c9ac614fc4051a768777cad3fbaf71cbd486a31540d8195db6a98",
        url = "https://developer.nvidia.com/isaac/download/third_party/opencv-3-3-1-x86_64-18-20190327c-tgz",
        type = "tgz",
        licenses = ["https://opencv.org/license.html"],
    )

    isaac_new_http_archive(
        name = "opencv_aarch64_jetpack42",
        build_file = clean_dep("//third_party:opencv_jetpack42.BUILD"),
        sha256 = "f50be71f870a5e064b859fd110a73eb075b08b6365847d2fc19c59c2bdebef91",
        url = "https://developer.nvidia.com/isaac/download/third_party/opencv_jetpack_4_2_b150_aarch64_nano-tar-xz",
        type = "tar.xz",
        licenses = ["https://opencv.org/license.html"],
    )

    isaac_new_http_archive(
        name = "libtensorflow_x86_64",
        build_file = clean_dep("//third_party:libtensorflow_x86_64.BUILD"),
        sha256 = "0f8072830081192ccb71ce5c80b43ea516d4a31be4aefe6fdc03a46a96d266fa",
        url = "https://developer.nvidia.com/isaac/download/third_party/libtensorflow_1_15_0-tar-gz",
        type = "tar.gz",
        licenses = ["https://raw.githubusercontent.com/tensorflow/tensorflow/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "libtensorflow_aarch64_jetpack43",
        build_file = clean_dep("//third_party:libtensorflow_aarch64_jetpack43.BUILD"),
        sha256 = "14b38de6fdb024185ec27f61bc6e7ef94c70ce677c6f93b53b8557e16a8c7b2d",
        url = "https://developer.nvidia.com/isaac/download/third_party/libtensorflow_1_15_0_jp43b79-tar-gz",
        type = "tar.gz",
        licenses = ["https://raw.githubusercontent.com/tensorflow/tensorflow/master/LICENSE"],
    )

    # libtorch for x86_64
    isaac_new_http_archive(
        name = "libtorch_x86_64",
        build_file = clean_dep("//third_party:libtorch_x86_64.BUILD"),
        sha256 = "203ffa86773e5e061ff249345012a23c8acc3feb3f68b93bd2aecbd9ba41c4ae",
        url = "https://developer.nvidia.com/isaac/download/third_party/libtorch_x86_64_1-1-0-v2-tar-xz",
        type = "tar.xz",
        licenses = ["https://github.com/pytorch/pytorch/blob/master/LICENSE"],
    )

    # libtorch for aarch64_jetpack42
    isaac_new_http_archive(
        name = "libtorch_aarch64_jetpack42",
        build_file = clean_dep("//third_party:libtorch_aarch64_jetpack42.BUILD"),
        sha256 = "3a66d995cd0b7254e82549edc1c09d2a5562f0fe186bb69c5855e3da7ab9f7d0",
        url = "https://developer.nvidia.com/isaac/download/third_party/libtorch_aarch64_jetpack42_1-1-0-v0-tar-gz",
        type = "tar.gz",
        licenses = ["https://github.com/pytorch/pytorch/blob/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "mobilenetv2",
        build_file = clean_dep("//third_party:mobilenetv2.BUILD"),
        sha256 = "a20d0c8d698502dc6a620528871c97a588885df7737556243a3412b39fce85e0",
        url = "https://developer.nvidia.com/isaac/download/third_party/mobilenetv2-1-4-224-tgz",
        type = "tgz",
        licenses = ["https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/mobilenet/mobilenet.py"],
    )

    isaac_new_http_archive(
        name = "mobilenetv2_onnx",
        build_file = clean_dep("//third_party:mobilenetv2_onnx.BUILD"),
        sha256 = "8ce2930074b6025c141fcfee9e2c63bb7183f5f19e27695931ce763956cab098",
        url = "https://rdk-public.s3.amazonaws.com/test_data/mobilenetv2-1_0_onnx.tar.xz",
        type = "tar.xz",
        licenses = ["https://raw.githubusercontent.com/onnx/models/master/models/image_classification/mobilenet/README.md"],
    )

    isaac_new_http_archive(
        name = "ml_test_data",
        build_file = clean_dep("//third_party:ml_test_data.BUILD"),
        sha256 = "2916fe0330ed1c2392148fe1ba8f8353ae3b694aa1c50d28d8f3df8f318ad57e",
        url = "https://developer.nvidia.com/isaac/download/third_party/ml_test_data_1_3-tar-xz",
        type = "tar.xz",
        licenses = ["//:LICENSE"],
    )

    # Source: TensorRT/v6.0/6.0.1.5-cl27267773-eed615fe/10.0-r400/Ubuntu18_04-x64/deb
    isaac_new_http_archive(
        name = "tensorrt_x86_64",
        build_file = clean_dep("//third_party:tensorrt_x86_64.BUILD"),
        sha256 = "e781ccb0bbe6dae9c73b67b225e1657770f908fb7aa9a044be929ab40652cfe1",
        url = "https://developer.nvidia.com/isaac/download/third_party/tensorrt_6-0-1-5-1+cuda10-0_amd64-ubuntu18_04-x64-deb-tar-xz",
        type = "tar.xz",
        licenses = ["https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html"],
    )

    # Source: SDKManager/JetPack_SDKs/4.3/L4T/78_19316_27599411/JETPACK_43_b78/NoDLA
    isaac_new_http_archive(
        name = "tensorrt_aarch64_jetpack43",
        build_file = clean_dep("//third_party:tensorrt_jetpack43.BUILD"),
        sha256 = "b40a63a14084acc6602925bf361d979615c5c975823c24ff1be94a9227631546",
        url = "https://developer.nvidia.com/isaac/download/third_party/tensorrt_6-0-1-9-1+cuda10-0_arm64-jetpack_43_b78-nodla-tar-xz",
        type = "tar.xz",
        licenses = ["https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html"],
    )

    # Source: SDKManager/JetPack_SDKs/4.3/L4T/78_19316_27599411/JETPACK_43_b78/DLA
    isaac_new_http_archive(
        name = "tensorrt_aarch64_jetpack42_dla",
        build_file = clean_dep("//third_party:tensorrt_jetpack42.BUILD"),
        sha256 = "da01870cb65b4c50e07049749bf5b04456f087755ea673e12df12fc796b49132",
        url = "https://developer.nvidia.com/isaac/download/third_party/tensorrt_6-0-1-9-1+cuda10-0_arm64-jetpack_43_b78-dla-tar-xz",
        type = "tar.xz",
        licenses = ["https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html"],
    )

    isaac_new_http_archive(
        name = "yolo_pretrained_models",
        build_file = clean_dep("//third_party:yolo_pretrained_models.BUILD"),
        sha256 = "30e674bcc7e2de4ac32ce815bf835126b4b48c01dde03bf7538404f50c47e606",
        url = "https://developer.nvidia.com/isaac/download/third_party/yolo_pretrained_models-tar-xz",
        type = "tar.xz",
        licenses = [
            "https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html",
            "https://raw.githubusercontent.com/pjreddie/darknet/master/LICENSE",
        ],
    )

    isaac_new_http_archive(
        name = "yolo_tensorrt_test_data",
        build_file = clean_dep("//third_party:yolo_tensorrt_test_data.BUILD"),
        sha256 = "917b720f579ea392bdc6ebe063e50faf702a494b4c5ab1ef7071b572463ee35e",
        url = "https://developer.nvidia.com/isaac/download/third_party/yolo_tensorrt_test_data-2018-12-tar-gz",
        type = "tar.gz",
        strip_prefix = "yolo_tensorrt_test_data_v2",
        licenses = [
            "https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html",
            "https://raw.githubusercontent.com/pjreddie/darknet/master/LICENSE",
        ],
    )

    isaac_new_http_archive(
        name = "yolo_tensorrt_lib",
        build_file = clean_dep("//third_party:yolo_tensorrt_lib.BUILD"),
        sha256 = "c7d346a536adf6ee7596a4c098c4f88793a132864379dc27339060e7f2808bb0",
        url = "https://developer.nvidia.com/isaac/download/third_party/yolo_library_20191125_anchors_fp16_tar_xz",
        type = "tar.xz",
        strip_prefix = "yolo_library_20191125_anchors_fp16",
        licenses = [
            "@yolo_tensorrt_lib//:LICENSE.md",
            "https://docs.nvidia.com/deeplearning/sdk/tensorrt-sla/index.html",
            "https://github.com/pjreddie/darknet/blob/master/LICENSE",
        ],
    )

    isaac_new_http_archive(
        name = "redtail",
        build_file = clean_dep("//third_party:redtail.BUILD"),
        sha256 = "a25fa2b181606f781220fcc22945ddb483d9d00fe093f113c4e79abb3e556013",
        url = "https://developer.nvidia.com/isaac/download/third_party/redtail-20190625-cc2745047cf5a0964bdd3a38fc8e851491e48e75-zip",
        type = "zip",
        strip_prefix = "redtail-cc2745047cf5a0964bdd3a38fc8e851491e48e75",
        licenses = ["@redtail//:LICENSE.md"],
    )

    isaac_new_http_archive(
        name = "tacotron2_model",
        build_file = clean_dep("//third_party:tacotron2_model.BUILD"),
        sha256 = "ffb88e4734700521925fec5926a4b29336b261368f1b02d2d61f5bb3f6d95d40",
        url = "https://developer.nvidia.com/isaac/download/third_party/tacotron2_streaming_fp32-v1-tar-gz",
        type = "tar.gz",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "waveglow_model",
        build_file = clean_dep("//third_party:waveglow_model.BUILD"),
        sha256 = "a3a08e91470f8870a56e4fc4ff6fe479c31797f8d846200958f77733fa1d6cbb",
        url = "https://developer.nvidia.com/isaac/download/third_party/waveglow_randVect_noiseTrunc_fp16-v0-tar-gz",
        type = "tar.gz",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "libargus_aarch64_nano",
        build_file = clean_dep("//third_party:libargus_nano.BUILD"),
        sha256 = "b989ef88992bdccc99072031225e76d0b84d792be785bcfb54584fd0f2ec7093",
        url = "https://developer.nvidia.com/isaac/download/third_party/libargus-2019-3-aarch64_nano-tgz",
        type = "tgz",
        strip_prefix = "libargus",
        licenses = ["https://raw.githubusercontent.com/pauldotknopf/JetsonTX1Drivers/master/nv_tegra/LICENSE.libargus"],
    )

    isaac_new_http_archive(
        name = "hgmm_impl",
        sha256 = "c5243683e164eb84da59702f4ac9f06794f795c79d3c335a174abfb7318af8b6",
        url = "https://developer.nvidia.com/isaac/download/third_party/libhgmm_impl_bionic_08_09_2019-tar-xz",
        build_file = clean_dep("//third_party:libhgmm_impl.BUILD"),
        type = "tar.xz",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "livox_sdk",
        sha256 = "1c62b3f85a548183100cc94730926b64df2feeb10883f7d3bd245708ef0340a5",
        url = "https://developer.nvidia.com/isaac/download/third_party/Livox-SDK-1-0-0-tar-gz",
        build_file = clean_dep("//third_party:livox_sdk.BUILD"),
        type = "tar.gz",
        strip_prefix = "Livox-SDK-1.0.0",
        licenses = ["//:LICENSE"],
    )

    isaac_new_git_repository(
        name = "lfll",
        remote = "https://github.com/nicopauss/LFLL.git",
        commit = "9d29453368c432e373acf712d51515505ef057b0",
        build_file = clean_dep("//third_party:lfll.BUILD"),
        patches = [clean_dep("//third_party:lfll.patch")],
        licenses = ["https://github.com/nicopauss/LFLL/blob/master/LICENSE"],
    )

    isaac_new_git_repository(
        name = "efll",
        remote = "https://github.com/zerokol/eFLL.git",
        commit = "640b8680b6535768f318172b0a28a5e4091d8f60",
        build_file = clean_dep("//third_party:efll.BUILD"),
        licenses = ["https://github.com/zerokol/eFLL/blob/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "object_pose_estimation_aae",
        build_file = clean_dep("//third_party:object_pose_estimation_aae.BUILD"),
        sha256 = "c967602105524862f2e075112d751a64dd61ad433ba8ed21905d36c91ae769a8",
        url = "https://developer.nvidia.com/isaac/download/third_party/object_pose_estimation_aae_v2-tar-gz",
        type = "tar.gz",
        strip_prefix = "AugmentedAutoencoder-master",
        licenses = ["https://github.com/DLR-RM/AugmentedAutoencoder/blob/master/LICENSE"],
    )

    isaac_new_http_archive(
        name = "path_segmentation_images",
        build_file = clean_dep("//third_party:path_segmentation_images.BUILD"),
        sha256 = "da9e7e16613bd480290c7491373922e8dde247f255eee201f583c82c601a453c",
        url = "https://developer.nvidia.com/isaac/download/third_party/path_segmentation_images_2019_11_14-tar-xz",
        type = "tar.xz",
        strip_prefix = "path_segmentation_images_2019_11_14",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "path_segmentation_logs",
        build_file = clean_dep("//third_party:path_segmentation_logs.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/path_segmentation_logs_2019_11_14-tar-xz",
        sha256 = "2e16fcf7bf922f933aec23ae879f59ed51f88b29f34d7316b663ee470bf7b94e",
        type = "tar.xz",
        strip_prefix = "path_segmentation_logs_2019_11_14",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "path_segmentation_pretrained_models",
        build_file = clean_dep("//third_party:path_segmentation_pretrained_models.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/path_segmentation_pretrained_models_2_2019_11_14-tar-xz",
        sha256 = "0278861657c710d48a9a1ae75378e67612e9f0cae1e1ef49d4c422593f5f3c96",
        type = "tar.xz",
        strip_prefix = "path_segmentation_pretrained_models",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "object_detection_test_data",
        build_file = clean_dep("//third_party:object_detection_test_data.BUILD"),
        sha256 = "e1cc108d997050473888efa59eccb0048f028518ccf627c8e9f6498e66ebfec1",
        url = "https://developer.nvidia.com/isaac/download/third_party/object_detection_evaluation_data_599c193-tar-gz",
        type = "tar.gz",
        strip_prefix = "object_detection_evaluation_data",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "sidewalk_segmentation_test_data",
        build_file = clean_dep("//third_party:sidewalk_segmentation_test_data.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/sidewalk_segmentation_test_data_20191018-tar-xz",
        sha256 = "1d9c1f268d6b779d23251aea8ccadba4cf1882d4a4370edc01f1c478988ca888",
        type = "tar.xz",
        strip_prefix = "sidewalk_segmentation_test_data",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "tlt_parser",
        build_file = clean_dep("//third_party:tlt_parser.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/tlt_parser_20191121-zip",
        sha256 = "1205669b58d78a93b9f5da5656682ffd2b7146014d7501448638d5c1f08ab1a7",
        type = "zip",
        strip_prefix = "tlt_parser",
        licenses = ["//:LICENSE"],
    )

    isaac_new_http_archive(
        name = "crop_and_downsample_test_data",
        build_file = clean_dep("//third_party:crop_and_downsample_test_data.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/crop_and_downsample_test_images_2019_11_20-tar-xz",
        sha256 = "0b7f834fd3be3ed93a823d0c6b480d75482eb3213d8d8b20c0d3776559cc3d91",
        type = "tar.xz",
        strip_prefix = "crop_and_downsample_test_images_2019_11_20",
        licenses = ["//:LICENSE"],
    )

    # GStreamer is a pipeline-based multimedia framework that links together a wide variety of
    # media processing systems. It is a common Linux system component. The host system provides
    # access to the libraries with a variety of licenses depending on your usage. You should
    # review licenses per your usage of GStreamer components.
    isaac_new_local_repository(
        name = "gstreamer",
        build_file = clean_dep("//third_party:gstreamer.BUILD"),
        path = "/usr/include/gstreamer-1.0",
        licenses = ["https://gitlab.freedesktop.org/gstreamer/gstreamer/blob/master/COPYING"],
    )

    # GLib is a set of low-level libraries useful for providing data structure handling for C,
    # portability wrappers, execution loops, and interfaces. It is a common Linux system
    # component. The host system provides access to the libraries.
    isaac_new_local_repository(
        name = "glib",
        build_file = clean_dep("//third_party:glib.BUILD"),
        path = "/usr/include/glib-2.0",
        licenses = ["https://github.com/GNOME/glib/blob/master/COPYING"],
    )

    isaac_new_local_repository(
        name = "glib_config",
        build_file = clean_dep("//third_party:glib_config.BUILD"),
        path = "/usr/lib/x86_64-linux-gnu/glib-2.0/include",
        licenses = ["https://github.com/GNOME/glib/blob/master/COPYING"],
    )

    # Precompiled AArch64 Jetson DeepStream module library to allow cross-compilation without
    # libraries installation on host.
    isaac_new_http_archive(
        name = "libdeepstream_module_aarch64_jetpack43",
        build_file = clean_dep("//third_party:libdeepstream_module_aarch64_jetpack43.BUILD"),
        url = "https://developer.nvidia.com/isaac/download/third_party/libdeepstream_module_aarch64_jetpack43_v4-tar-xz",
        sha256 = "db914ee819ccce83cc81b5d5a79bbd9667f0f9f445ff6ecb51133955590f775d",
        type = "tar.xz",
        licenses = ["//:LICENSE"],
    )
