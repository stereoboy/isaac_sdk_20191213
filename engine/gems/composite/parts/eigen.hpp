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
#include <array>

#include "engine/core/math/types.hpp"
#include "engine/core/string_literal.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// Specialization of PartTraits for Eigen::Vector
template <typename K, int N>
struct PartTraits<Eigen::Matrix<K, N, 1>> {
  using Scalar = K;
  using Type = Eigen::Matrix<K, N, 1>;

  static_assert(N > 0, "Only fixed-size vectors are supported");
  static constexpr size_t kElementCount = N;

  static Type CreateFromScalars(const K* scalars) {
    Type result;
    std::copy(scalars, scalars + N, &result[0]);
    return result;
  }

  static void WriteToScalars(const Type& value, K* scalars) {
    const K* source = &value[0];
    std::copy(source, source + N, scalars);
  }
};

}  // namespace isaac
