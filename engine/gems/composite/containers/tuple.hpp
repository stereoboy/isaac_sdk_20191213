/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <tuple>

#include "engine/gems/composite/part_list.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

// A type indicating that a composite will use an std::tuple to store data.
//
// This container is based on an std::tuple. It gives efficient access to parts via a compile-time
// integer without packing / unpacking from scalars. It currently does not support scalar access.
struct TupleCompositeContainer {};

// Container traits for the EigenVectorCompositeContainer
template <typename PartListT>
struct CompositeContainerTraits<PartListT, TupleCompositeContainer> {
  using Container = typename PartListT::Tuple;

  template <size_t PartIndex>
  static auto GetPart(const Container& container) {
    return std::get<PartIndex>(container);
  }

  template <size_t PartIndex>
  static auto& GetPart(Container& container) {
    return std::get<PartIndex>(container);
  }

  template <size_t ElementIndex>
  static auto GetScalar(const Container& container) {
    static_assert(ElementIndex >= 0, "not implemented");
  }

  template <size_t ElementIndex>
  static auto& GetScalar(Container& container) {
    static_assert(ElementIndex >= 0, "not implemented");
  }

  // TODO Scalar access might be added in a future version.
  // static K GetScalar(const Container& container, size_t index) { }
  // static K& GetScalar(Container& container, size_t index) { }
};

}  // namespace isaac
