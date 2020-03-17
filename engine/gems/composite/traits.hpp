/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

namespace isaac {

// Traits to make various containers compatible with composite types.
template <typename PartListT, typename Container>
struct CompositeContainerTraits;

// Traits to make various math types compatible to be used as parts in composites.
//
// The necessary elements are:
//   // The scalar type used by the part
//   using Scalar = ???;
//   // The type of the part
//   using Type = ???;
//   // The number of scalars used by the part
//   static constexpr size_t kElementCount = N;
//   // Creates the part from a list of scalars
//   static Type CreateFromScalars(const K* scalars);
//   // Writes the type to a list of scalars
//   static void WriteToScalars(const Type& value, K* scalars);
template <typename T, typename Enable = void>
struct PartTraits;

// A handle type which can be used in a part list to use a composite as a part itself
template <template<typename> typename Child>
struct CompositeAsPart {};

template <typename T, typename ContainerTag>
struct ActualType {
  using type = T;
};

template <template <typename> typename Child, typename ContainerTag>
struct ActualType<CompositeAsPart<Child>, ContainerTag> {
  using type = Child<ContainerTag>;
};

template <typename T, typename ContainerTag>
using ActualType_t = typename ActualType<T, ContainerTag>::type;

}  // namespace isaac
