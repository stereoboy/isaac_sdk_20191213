/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "blob.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "engine/core/assert.hpp"

namespace isaac {
namespace serialization {

size_t AccumulateLength(const std::vector<ByteArrayConstView>& blobs) {
  size_t length = 0;
  for (const auto& blob : blobs) {
    length += blob.size();
  }
  return length;
}

byte* CopyAll(const std::vector<ByteArrayConstView>& blobs, byte* dst, byte* dst_end) {
  for (const auto& blob : blobs) {
    std::copy(blob.begin(), blob.end(), dst);
    dst += blob.size();
    ASSERT(dst <= dst_end, "Out of bounds");
  }
  return dst;
}

void BlobsToLengths32u(const std::vector<ByteArrayConstView>& blobs,
                       std::vector<uint32_t>& lengths) {
  lengths.resize(blobs.size());
  for (size_t i = 0; i < blobs.size(); i++) {
    const size_t length = blobs[i].size();
    ASSERT(length <= std::numeric_limits<uint32_t>::max(),
           "Blob to big to store its length as a 32-bit unsigned (length: %zu)", length);
    lengths[i] = static_cast<uint32_t>(length);
  }
}

void CapnpArraysToBlobs(const kj::ArrayPtr<const kj::ArrayPtr<const ::capnp::word>> segments,
                        std::vector<ByteArrayConstView>& blobs) {
  blobs.resize(segments.size());
  for (size_t i = 0; i < blobs.size(); i++) {
    auto segment = segments[i].asBytes();
    blobs[i] = ByteArrayConstView{segment.begin(), segment.size()};
  }
}

}  // namespace serialization
}  // namespace isaac
