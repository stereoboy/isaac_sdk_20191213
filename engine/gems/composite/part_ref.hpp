/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/gems/composite/traits.hpp"

namespace isaac {

// Provides reference support for parts when they are stored in a flat array
template <typename T>
class PartRef {
 public:
  using Scalar = typename PartTraits<T>::Scalar;

  PartRef(const PartRef&) = delete;
  void operator=(const PartRef&) = delete;

  PartRef(PartRef&&) = default;
  void operator=(PartRef&&) = delete;

  // Creates reference backed by given scalar pack
  PartRef(Scalar* scalars) : scalars_(scalars) {}

  // Writes a part to scalars using part traits
  template <typename Other>
  void operator=(const Other& value) {
    PartTraits<T>::WriteToScalars(value, scalars_);
  }

  T get() const {
    return PartTraits<T>::CreateFromScalars(scalars_);
  }

  operator T() const {
    return get();
  }

 private:
  Scalar* scalars_;
};

// Similar to PartRef, but provides only const access
template <typename T>
class PartConstRef {
 public:
  using Scalar = typename PartTraits<T>::Scalar;

  PartConstRef(const PartConstRef&) = delete;
  void operator=(const PartConstRef&) = delete;

  PartConstRef(PartConstRef&&) = default;
  void operator=(PartConstRef&&) = delete;

  PartConstRef(const Scalar* scalars) : scalars_(scalars) {}

  T get() const {
    return PartTraits<T>::CreateFromScalars(scalars_);
  }

  operator T() const {
    return get();
  }

 private:
  const Scalar* scalars_;
};

// Evaluates a part ref to the corresponding type (creating a copy)
template <typename T>
T Evaluate(const PartRef<T>& ref) {
  return ref.get();
}

// Evaluates a part ref to the corresponding type (creating a copy)
template <typename T>
T Evaluate(const PartConstRef<T>& ref) {
  return ref.get();
}

// Evaluates a part to the corresponding type by return a reference to the original value
template <typename T>
const T& Evaluate(const T& value) {
  return value;
}

}  // namespace isaac
