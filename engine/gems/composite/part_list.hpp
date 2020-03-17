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

#include "engine/gems/composite/traits.hpp"

namespace isaac {

// A compile-time list of types
template <typename... Parts>
struct PartList {
  static constexpr size_t kNumParts = sizeof...(Parts);
  using Tuple = std::tuple<Parts...>;
};

namespace detail {

// Helper to add a type to a template type.
template <class T> struct AddType { using type = T; };

// Helper to a add a value to a template type. It uses the enum trick to avoid all kinds of trouble
// around static constexpr *sigh*
template <size_t I> struct AddValue { enum : size_t { value = I }; };

template <size_t N, typename T, typename PL>
struct HeadNImpl;

template <size_t N, typename... As, typename X, typename... Bs>
struct HeadNImpl<N, PartList<As...>, PartList<X, Bs...>>
: HeadNImpl<N - 1, PartList<As..., X>, PartList<Bs...>> {};

template <typename... As, typename X, typename... Bs>
struct HeadNImpl<0, PartList<As...>, PartList<X, Bs...>> : AddType<PartList<As...>> {};

template <typename... As>
struct HeadNImpl<0, PartList<As...>, PartList<>> : AddType<PartList<As...>> {};

// First N parts of part list
template <size_t N, typename PL>
using HeadN = typename HeadNImpl<N, PartList<>, PL>::type;

template <typename... T>
struct BackImpl;

template <typename X>
struct BackImpl<X> : AddType<X> {};

template <typename X, typename... As>
struct BackImpl<X, As...> : AddType<typename BackImpl<As...>::type> {};

template <typename... Parts>
struct BackImpl<PartList<Parts...>> : detail::AddType<typename BackImpl<Parts...>::type> {};

// The last part of a part list
template <typename PL>
using Back = typename BackImpl<PL>::type;

// The N-th part of a part list
template <size_t N, typename PL>
using At = typename std::tuple_element<N, typename PL::Tuple>::type;
// using At = Back<HeadN<N + 1, PL>>;

}  // namespace detail

// Number of elements in a part list as define by part traits
template <typename... T>
struct Length;

template <>
struct Length<> : detail::AddValue<0> {};

struct EigenVectorCompositeContainer;
template <template<typename> typename Child>
struct Length<CompositeAsPart<Child>>
    : detail::AddValue<Child<EigenVectorCompositeContainer>::kElementCount> {};

template <typename T>
struct Length<T> : detail::AddValue<PartTraits<T>::kElementCount> {};

template <typename Head, typename... Tail>
struct Length<Head, Tail...> : detail::AddValue<Length<Head>::value + Length<Tail...>::value> {};

template <typename... Parts>
struct Length<PartList<Parts...>> : detail::AddValue<Length<Parts...>::value> {};

// Index of the first element of a part in a part list
template <int PI, typename PL>
struct ElementStartIndex : detail::AddValue<Length<detail::HeadN<PI, PL>>::value> {};

// The scalar type of a part list
template <typename PartListT, typename ContainerTag>
using PartListScalar
    = typename PartTraits<ActualType_t<detail::At<0, PartListT>, ContainerTag>>::Scalar;

}  // namespace isaac
