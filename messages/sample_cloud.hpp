/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once
#include <algorithm>
#include <utility>
#include <vector>

#include "engine/core/array/byte_array.hpp"
#include "engine/core/buffers/shared_buffer.hpp"
#include "engine/core/logger.hpp"
#include "engine/core/sample_cloud/sample_cloud.hpp"
#include "engine/gems/math/float16.hpp"
#include "messages/element_type.hpp"
#include "messages/sample_cloud.capnp.h"

template <typename K, size_t Channels, typename BufferType>
bool FromProto(::SampleCloudProto::Reader reader,
               const std::vector<isaac::SharedBuffer>& buffers,
               isaac::SampleCloudBase<K, Channels, BufferType>& sample_cloud_view) {
  static_assert(
      !isaac::BufferTraits<BufferType>::kIsMutable && !isaac::BufferTraits<BufferType>::kIsOwning,
      "Invalid buffer type. Buffer cannot own or mutate data");
  static_assert(isaac::BufferTraits<BufferType>::kStorageMode == isaac::BufferStorageMode::Host ||
                    isaac::BufferTraits<BufferType>::kStorageMode == isaac::BufferStorageMode::Cuda,
                "Unknown buffer storage mode.");

  // Check element type
  const isaac::ElementType message_element_type = FromProto(reader.getElementType());
  const isaac::ElementType expected_element_type = isaac::GetElementType<K>();
  if (expected_element_type != message_element_type) {
    LOG_ERROR("Sample Cloud element type does not match: actual=%d, expected=%d",
              message_element_type, expected_element_type);
    return false;
  }

  if (Channels != reader.getChannels()) {
    LOG_ERROR("Sample Cloud Channel counts do not match: actual = %d, expected = %d",
              reader.getChannels(), Channels);
    return false;
  }
  auto storage_order = reader.getStorageOrder();
  if (storage_order != ::SampleCloudProto::StorageOrder::INTERLEAVED) {
    LOG_ERROR("Storage Orders do not match");
    return false;
  }

  // Get buffer
  const size_t buffer_index = reader.getDataBufferIndex();
  if (buffer_index >= buffers.size()) {
    LOG_ERROR("Buffer index %u out of range (%zu): ", buffer_index, buffers.size());
    return false;
  }
  const isaac::SharedBuffer& buffer = buffers[buffer_index];

  // Create view
  const size_t sample_count = reader.getSampleCount();
  const auto source_buffer_view = buffer.const_view<BufferType>();
  if (source_buffer_view.size() != Channels * sample_count * sizeof(K)) {
    LOG_ERROR("Buffer does not have enough storage");
    return false;
  }
  sample_cloud_view = isaac::SampleCloudBase<K, Channels, BufferType>(
      {source_buffer_view.begin(), Channels * sample_count * sizeof(K)},
      sample_count);
  return true;
}

template <typename K, size_t Channels, typename BufferType>
void ToProto(isaac::SampleCloudBase<K, Channels, BufferType> sample_cloud,
             ::SampleCloudProto::Builder builder,
             std::vector<isaac::SharedBuffer>& buffers) {
  static_assert(isaac::BufferTraits<BufferType>::kIsOwning, "Must own the data to send it.");
  static_assert(isaac::BufferTraits<BufferType>::kStorageMode == isaac::BufferStorageMode::Host ||
                    isaac::BufferTraits<BufferType>::kStorageMode == isaac::BufferStorageMode::Cuda,
                "Unknown buffer storage mode.");
  // Set the element type
  builder.setElementType(ToProto(isaac::GetElementType<K>()));
  // Set the sample cloud meta data
  builder.setChannels(Channels);
  builder.setSampleCount(sample_cloud.size());
  builder.setStorageOrder(::SampleCloudProto::StorageOrder::INTERLEAVED);
  // Store the sample cloud data as a buffer
  builder.setDataBufferIndex(buffers.size());
  buffers.emplace_back(std::move(sample_cloud.data()));
}
