/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <type_traits>

#include "engine/core/string_literal.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// Specialization of PartTraits for scalars
template <typename K>
struct PartTraits<K, std::enable_if_t<std::is_floating_point<K>::value>> {
  using Scalar = K;

  static constexpr size_t kElementCount = 1;

  static K CreateFromScalars(const K* scalars) {
    return scalars[0];
  }

  static void WriteToScalars(K value, K* scalars) {
    scalars[0] = value;
  }
};

}  // namespace isaac
