/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/math/pose2.hpp"
#include "engine/core/math/so2.hpp"
#include "engine/core/math/types.hpp"
#include "engine/core/string_literal.hpp"
#include "engine/gems/composite/parts/eigen.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// Specialization of PartTraits for SO2
template <typename K>
struct PartTraits<SO2<K>> {
  using Scalar = K;

  static constexpr size_t kElementCount = 2;

  static SO2<K> CreateFromScalars(const K* scalars) {
    return SO2<K>::FromNormalized(scalars[0], scalars[1]);
  }

  static void WriteToScalars(const SO2<K>& value, K* scalars) {
    scalars[0] = value.cos();
    scalars[1] = value.sin();
  }
};

// Specialization of PartTraits for Pose2
template <typename K>
struct PartTraits<Pose2<K>> {
  using Scalar = K;
  using PartTraitsA = PartTraits<SO2<K>>;
  using PartTraitsB = PartTraits<Vector<K, 2>>;

  static constexpr size_t kElementCount =
      PartTraitsA::kElementCount + PartTraitsB::kElementCount;

  static Pose2<K> CreateFromScalars(const K* scalars) {
    return Pose2<K>{
        PartTraitsA::CreateFromScalars(scalars),
        PartTraitsB::CreateFromScalars(scalars + PartTraitsA::kElementCount)
    };
  }

  static void WriteToScalars(const Pose2<K>& value, K* scalars) {
    PartTraitsA::WriteToScalars(value.rotation, scalars);
    PartTraitsB::WriteToScalars(value.translation, scalars + PartTraitsA::kElementCount);
  }
};

}  // namespace isaac
