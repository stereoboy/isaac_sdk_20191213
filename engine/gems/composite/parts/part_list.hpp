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

#include "engine/gems/composite/part_ref.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// Part traits for composite types.
// TODO Make sure this really only is enabled for composites
template <typename T>
struct PartTraits<T, typename std::enable_if<T::kIsComposite, void>::type> {
  using Type = T;
  using Scalar = typename Type::Scalar;

  static constexpr size_t kElementCount = Type::kElementCount;

  static T CreateFromScalars(const Scalar* scalars) {
    // FIXME This does not work for all containers!
    T result;
    std::copy(scalars, scalars + kElementCount, result.data.data());
    return std::move(result);
  }

  static void WriteToScalars(const T& value, Scalar* scalars) {
    for (size_t i = 0; i < kElementCount; i++) {
      scalars[i] = T::TheCompositeContainerTraits::GetScalar(value.data, i);
    }
  }
};

}  // namespace isaac
