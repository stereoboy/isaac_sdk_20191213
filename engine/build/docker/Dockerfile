#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
# Full Isaac SDK build environment for development activities.
FROM nvidia/cudagl:10.0-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive

ADD scripts/install_dependencies.sh engine/build/scripts/
ADD scripts/requirements.txt engine/build/scripts/

RUN \
    apt update && \
    engine/build/scripts/install_dependencies.sh

# Install cuDNN v7 as described in https://hub.docker.com/r/kaixhin/cudnn/dockerfile
ENV ML_REPO_PKG=nvidia-machine-learning-repo-ubuntu1804_1.0.0-1_amd64.deb
RUN apt-get install wget
RUN wget http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/$ML_REPO_PKG && \
  dpkg -i $ML_REPO_PKG && \
  apt-get update && apt-get install -y libcudnn7 libcudnn7-dev
# Install AHA utility to prepare HTML reports from the test results
RUN  apt-get install -y aha rsync xz-utils

# Configure the build for CUDA configuration
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
