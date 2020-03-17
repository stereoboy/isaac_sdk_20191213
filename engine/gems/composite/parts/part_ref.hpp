/*
Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/gems/composite/part_ref.hpp"
#include "engine/gems/composite/traits.hpp"

namespace isaac {

template <typename T>
struct PartTraits<PartRef<T>> : PartTraits<T> {};

template <typename T>
struct PartTraits<PartConstRef<T>> : PartTraits<T> {};

}  // namespace isaac
