/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/array/cpu_array.hpp"
#include "engine/core/math/types.hpp"
#include "engine/gems/composite/part_list.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// A type indicating that a composite will use a MemoryView to store data.
//
// This container stores all parts as a packed list of scalars. It allows efficient
// access to scalars via a compile-time or run-time index. When parts are accessed scalars are
// automatically packed or unpacked.
//
// A composite using this container will not hold any memory on construction. Thus accessing parts
// is undefined behavior until the memory view container is properly initialized.
struct MemoryViewCompositeContainer {};

// Container traits for the MemoryViewCompositeContainer
template <typename PartListT>
struct CompositeContainerTraits<PartListT, MemoryViewCompositeContainer> {
  using Scalar = PartListScalar<PartListT, MemoryViewCompositeContainer>;
  using Container = CpuArrayView<Scalar>;

  template <size_t PartIndex>
  static auto GetPart(const Container& container) {
    return PartTraits<detail::At<PartIndex, PartListT>>::CreateFromScalars(
        container.begin() + ElementStartIndex<PartIndex, PartListT>::value);
  }

  template <size_t PartIndex>
  static auto GetPart(Container& container) {
    return PartRef<detail::At<PartIndex, PartListT>>(
        container.begin() + ElementStartIndex<PartIndex, PartListT>::value);
  }

  template <size_t ElementIndex>
  static Scalar GetScalar(const Container& container) {
    return container.begin()[ElementIndex];
  }

  template <size_t ElementIndex>
  static Scalar& GetScalar(Container& container) {
    return container.begin()[ElementIndex];
  }

  static Scalar GetScalar(const Container& container, size_t index) {
    return container.begin()[index];
  }

  static Scalar& GetScalar(Container& container, size_t index) {
    return container.begin()[index];
  }
};

}  // namespace isaac
