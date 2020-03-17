/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/buffers/algorithm.hpp"
#include "engine/core/sample_cloud/sample_cloud.hpp"

namespace isaac {

// Copy tensors with compatibale memory layouts
template <typename K, size_t Channels, typename SourceContainer, typename TargetContainer>
void Copy(const SampleCloudBase<K, Channels, SourceContainer>& source,
          SampleCloudBase<K, Channels, TargetContainer>& target) {
  ASSERT(source.size() == target.size(), "Sample clouds count mismatch");
  ASSERT(source.data().size() == target.data().size(), "Sample cloud buffer size mismatch");
  CopyArrayRaw(source.data().begin(), BufferTraits<SourceContainer>::kStorageMode,
               target.data().begin(), BufferTraits<TargetContainer>::kStorageMode,
               source.data().size());
}

}  // namespace isaac
