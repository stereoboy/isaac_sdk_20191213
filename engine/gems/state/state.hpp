/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/math/types.hpp"

namespace isaac {
namespace state {

// A state element represented by an array of floating points
template<typename K, size_t N>
struct State {
  using Scalar = K;
  static constexpr size_t kDimension = N;
  Vector<K, N> elements;
};

// Special case of State for dimension = 0
template<typename K>
struct State<K, 0> {
  using Scalar = K;
  static constexpr size_t kDimension = 0;
};

// Used together with State to define state. See documentation of State
#define ISAAC_STATE_VAR(INDEX, NAME) \
  const auto NAME() const { \
    static_assert(INDEX < this->kDimension, "Index out of range"); \
    return this->elements[INDEX]; \
  } \
  auto& NAME() { \
    static_assert(INDEX < this->kDimension, "Index out of range"); \
    return this->elements[INDEX]; \
  } \
  enum { kI_##NAME = INDEX }

}  // namespace state
}  // namespace isaac
