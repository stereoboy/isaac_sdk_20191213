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
#include "engine/gems/composite/part_list.hpp"
#include "engine/gems/composite/parts/eigen.hpp"
#include "engine/gems/composite/parts/part_list.hpp"
#include "engine/gems/composite/parts/part_ref.hpp"
#include "engine/gems/composite/parts/pose.hpp"
#include "engine/gems/composite/parts/scalar.hpp"
#include "engine/gems/composite/traits.hpp"

// A macro which has to be used inside a typelist composite to define some required types and
// functions. It also provides `Scalar`, the type of scalars, and `kElementCount`, the number of
// elements in the composite.
#define ISAAC_COMPOSITE_BASE(PARTS, CONTAINER_TAG)                                                 \
  using ThePartList = PARTS;                                                                       \
  using TheContainerTag = CONTAINER_TAG;                                                           \
  using TheCompositeContainerTraits =                                                              \
      ::isaac::CompositeContainerTraits<ThePartList, TheContainerTag>;                             \
  using Scalar = ::isaac::PartListScalar<PARTS, CONTAINER_TAG>;                                    \
  static constexpr size_t kElementCount = ::isaac::Length<PARTS>::value;                           \
  template <int kIndex, typename Fake = void> struct PartName;                                     \
  template <int kIndex> constexpr ::isaac::string_literal part_name() {                            \
    return PartName<kIndex>::value;                                                                \
  }                                                                                                \
  template <int kIndex, typename Fake = void> struct ElementName;                                  \
  template <int kIndex> constexpr ::isaac::string_literal scalar_name() {                          \
    return ElementName<kIndex>::value;                                                             \
  }

namespace isaac {

// Composite of parts which is based on a list of types fixed at compile time.
//
// A composite consists of multiple parts, for example a double or a two-dimensional vector. The
// typelist composite type allows the user to define a type which contains multiple elements. This
// is similar to a struct, but has the big advantage that the composite type can be serialized
// into a flat list of scalars. It can also be used to construct the composite type back from that
// serialized list of parts.
//
// The composite can use different underlying container types, and the container type has a big
// influence on the usage of the composite. By using a TupleCompositeContainer the composite is
// serialized with an std::tuple which provides full access to parts, but incurs a runtime penalty
// for serialization or deserialization as a copy is required. On the other hand by using an
// array-based container like EigenVectorCompositeContainer or MemoryViewCompositeContainer
// the data already is available in a serialized forme, but needs to be copied when parts are
// accessed.
//
// Example usage:
//
//   1) Define the list of parts in the composite:
//   using MyParts = PartList<double, Pose2d, double>;
//
//   2) Define the composite state with the desired container type:
//   struct MyState : TypelistComposite<MyParts, EigenVectorCompositeContainer> {
//     ISAAC_COMPOSITE_BASE(MyParts, EigenVectorCompositeContainer);
//     ISAAC_COMPOSITE_PART_SCALAR(0, foo);
//     ISAAC_COMPOSITE_PART_POSE2(1, bar);
//     ISAAC_COMPOSITE_PART_SCALAR(2, zur);
//   };
//
//   This type will now provide member functions foo(), bar(), and zur() which can be used to write
//   and read parts in the composite.
template <typename PartListT, typename ContainerTag>
struct TypelistComposite {
  static constexpr bool kIsComposite = true;

  ISAAC_COMPOSITE_BASE(PartListT, ContainerTag);

  typename TheCompositeContainerTraits::Container data;

  template <size_t Index>
  auto part() const -> decltype(TheCompositeContainerTraits::template GetPart<Index>(data)) {
    return TheCompositeContainerTraits::template GetPart<Index>(data);
  }
  template <size_t Index>
  auto part() -> decltype(TheCompositeContainerTraits::template GetPart<Index>(data)) {
    return TheCompositeContainerTraits::template GetPart<Index>(data);
  }

  template <size_t Index>
  Scalar scalar() const {
    return TheCompositeContainerTraits::template GetScalar<Index>(data);
  }
  template <size_t Index>
  auto scalar() -> decltype(TheCompositeContainerTraits::template GetScalar<Index>(data)) {
    return TheCompositeContainerTraits::template GetScalar<Index>(data);
  }
};

}  // namespace isaac

// This macro is used inside a composite to name a scalar part. The type must match the part
// type at the corresponding index in the composite part list.
#define ISAAC_COMPOSITE_PART_SCALAR(kIndex, NAME)                                                  \
  using kT_##NAME = ::isaac::detail::At<kIndex, ThePartList>;                                      \
  using kPTraits_##NAME = ::isaac::PartTraits<kT_##NAME>;                                          \
  static_assert(std::is_same<Scalar, kPTraits_##NAME::Scalar>::value,                              \
                "Invalid type: the scalar type of the part at index " #kIndex " does not match."); \
  static_assert(std::is_same<Scalar, kT_##NAME>::value,                                            \
                "Invalid type: the state does not have a scalar at index " #kIndex);               \
  enum { kP_##NAME = kIndex };                                                                     \
  template <typename Fake> struct PartName<kIndex, Fake> {                                         \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  enum { kI_##NAME = ::isaac::ElementStartIndex<kIndex, ThePartList>::value };                     \
  template <typename Fake> struct ElementName<kI_##NAME, Fake> {                                   \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  auto NAME() const {                                                                              \
    return this->template part<kIndex>();                                                          \
  }                                                                                                \
  auto NAME() -> decltype(this->template part<kIndex>()) {                                         \
    return this->template part<kIndex>();                                                          \
  }

// This macro is used inside a composite to name a vector part. The type must match the part
// type at the corresponding index in the composite part list.
#define ISAAC_COMPOSITE_PART_VECTOR(kIndex, NAME)                                                  \
  using kT_##NAME = ::isaac::detail::At<kIndex, ThePartList>;                                      \
  using kPTraits_##NAME = ::isaac::PartTraits<kT_##NAME>;                                          \
  static_assert(std::is_same<Scalar, kPTraits_##NAME::Scalar>::value,                              \
                "Invalid type: the scalar type of the part at index " #kIndex " does not match."); \
  static_assert(std::is_same<Vector<Scalar, kT_##NAME::RowsAtCompileTime>, kT_##NAME>::value,      \
                "Invalid type: the state does not have a Vector at index " #kIndex);               \
  enum { kP_##NAME = kIndex };                                                                     \
  template <typename Fake> struct PartName<kIndex, Fake> {                                         \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  enum { kI_##NAME = ::isaac::ElementStartIndex<kIndex, ThePartList>::value };                     \
  /* TODO: add indices for vetor scalars */                                                        \
  /* TODO: add names for vetor scalars */                                                          \
  auto NAME() const {                                                                              \
    return this->template part<kIndex>();                                                          \
  }                                                                                                \
  auto NAME() -> decltype(this->template part<kIndex>()) {                                         \
    return this->template part<kIndex>();                                                          \
  }

// This macro is used inside a composite to name a Pose2 part. The type must match the part
// type at the corresponding index in the composite part list.
#define ISAAC_COMPOSITE_PART_POSE2(kIndex, NAME)                                                   \
  using kT_##NAME = ::isaac::detail::At<kIndex, ThePartList>;                                      \
  using kPTraits_##NAME = ::isaac::PartTraits<kT_##NAME>;                                          \
  static_assert(std::is_same<Scalar, kPTraits_##NAME::Scalar>::value,                              \
                "Invalid type: the scalar type of the part at index " #kIndex " does not match."); \
  static_assert(std::is_same<Pose2<Scalar>, kT_##NAME>::value,                                     \
                "Invalid type: the state does not have a Pose2 at index " #kIndex);                \
  enum { kP_##NAME = kIndex };                                                                     \
  template <typename Fake> struct PartName<kIndex, Fake> {                                         \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  enum { kI_##NAME = ::isaac::ElementStartIndex<kIndex, ThePartList>::value };                     \
  enum { kI_##NAME##_px = kI_##NAME };                                                             \
  enum { kI_##NAME##_py = kI_##NAME + 1 };                                                         \
  enum { kI_##NAME##_qx = kI_##NAME + 2 };                                                         \
  enum { kI_##NAME##_qy = kI_##NAME + 3 };                                                         \
  template <typename Fake> struct ElementName<kI_##NAME##_px, Fake> {                              \
    static constexpr ::isaac::string_literal value{#NAME "/px"};                                   \
  };                                                                                               \
  template <typename Fake> struct ElementName<kI_##NAME##_py, Fake> {                              \
    static constexpr ::isaac::string_literal value{#NAME "/py"};                                   \
  };                                                                                               \
  template <typename Fake> struct ElementName<kI_##NAME##_qx, Fake> {                              \
    static constexpr ::isaac::string_literal value{#NAME "/qx"};                                   \
  };                                                                                               \
  template <typename Fake> struct ElementName<kI_##NAME##_qy, Fake> {                              \
    static constexpr ::isaac::string_literal value{#NAME "/qy"};                                   \
  };                                                                                               \
  auto NAME() const {                                                                              \
    return this->template part<kIndex>();                                                          \
  }                                                                                                \
  auto NAME() -> decltype(this->template part<kIndex>()) {                                         \
    return this->template part<kIndex>();                                                          \
  }

// This macro is used inside a composite to name a sub composite part. The type must match the
// part type at the corresponding index in the composite part list.
#define ISAAC_COMPOSITE_PART_COMPOSITE(kIndex, NAME)                                               \
  using kT_##NAME = ActualType_t<::isaac::detail::At<kIndex, ThePartList>, TheContainerTag>;       \
  static_assert(std::is_same<Scalar, typename ::isaac::PartTraits<kT_##NAME>::Scalar>::value,      \
                "Invalid type: the scalar type of the part at index " #kIndex " does not match."); \
  enum { kP_##NAME = kIndex };                                                                     \
  template <typename Fake> struct PartName<kIndex, Fake> {                                         \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  enum { kI_##NAME = ::isaac::ElementStartIndex<kIndex, ThePartList>::value };                     \
  template <typename Fake> struct ElementName<kI_##NAME, Fake> {                                   \
    static constexpr ::isaac::string_literal value{#NAME};                                         \
  };                                                                                               \
  auto NAME() const {                                                                              \
    return this->template part<kIndex>();                                                          \
  }                                                                                                \
  auto NAME() -> decltype(this->template part<kIndex>()) {                                         \
    return this->template part<kIndex>();                                                          \
  }
