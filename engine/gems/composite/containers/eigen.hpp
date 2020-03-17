/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/core/math/types.hpp"
#include "engine/gems/composite/part_list.hpp"
#include "engine/gems/composite/part_ref.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// A type indicating that a composite will use an Eigen::Vector to store data.
//
// This container stores all parts as a packed list of scalars. It allows efficient
// access to scalars via a compile-time or run-time index. When parts are accessed scalars are
// automatically packed or unpacked.
//
// A composite using this container will store scalars on the stack in an eigen vector with
// size fixed at compile-time.
struct EigenVectorCompositeContainer {};

// Container traits for the EigenVectorCompositeContainer
template <typename PartListT>
struct CompositeContainerTraits<PartListT, EigenVectorCompositeContainer> {
  using Scalar = PartListScalar<PartListT, EigenVectorCompositeContainer>;
  using Container = Vector<Scalar, Length<PartListT>::value>;

  template <size_t PartIndex>
  static auto GetPart(const Container& container) {
    using type = ActualType_t<detail::At<PartIndex, PartListT>, EigenVectorCompositeContainer>;
    return PartTraits<type>::CreateFromScalars(
        container.data() + ElementStartIndex<PartIndex, PartListT>::value);
  }

  template <size_t PartIndex>
  static auto GetPart(Container& container) {
    using type = ActualType_t<detail::At<PartIndex, PartListT>, EigenVectorCompositeContainer>;
    return PartRef<type>(container.data() + ElementStartIndex<PartIndex, PartListT>::value);
  }

  template <size_t ElementIndex>
  static Scalar GetScalar(const Container& container) {
    return container[ElementIndex];
  }

  template <size_t ElementIndex>
  static Scalar& GetScalar(Container& container) {
    return container[ElementIndex];
  }

  static Scalar GetScalar(const Container& container, size_t index) {
    return container[index];
  }

  static Scalar& GetScalar(Container& container, size_t index) {
    return container[index];
  }
};

}  // namespace isaac
