/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <utility>
#include <vector>

#include "engine/core/tensor/tensor.hpp"
#include "engine/gems/state/state.hpp"
#include "messages/state.capnp.h"
#include "messages/tensor.hpp"

namespace isaac {
namespace state {

// Serializes a state into a StateProto message
template <typename K, size_t N>
void ToProto(const State<K, N>& state, StateProto::Builder proto,
             std::vector<isaac::SharedBuffer>& buffers) {
  // TODO Currently we only support double
  Tensor3d tensor(1, 1, static_cast<int>(N));
  for (size_t i = 0; i < N; i++) {
    tensor(0, 0, i) = state.elements[i];
  }
  ToProto(std::move(tensor), proto.initPack(), buffers);
}

template <typename State>
void ToProto(const std::vector<State>& state_series, StateProto::Builder proto,
             std::vector<isaac::SharedBuffer>& buffers) {
  // TODO Check that the template type State is actually of type State<K, N>
  constexpr size_t N = State::kDimension;
  // TODO Currently we only support double
  Tensor3d tensor(1, state_series.size(), static_cast<int>(N));
  for (size_t k = 0; k < state_series.size(); k++) {
    for (size_t i = 0; i < N; i++) {
      tensor(0, k, i) = state_series[k].elements[i];
    }
  }
  ToProto(std::move(tensor), proto.initPack(), buffers);
}

// Deserializes a state from a StateProto message
template <typename K, size_t N>
bool FromProto(StateProto::Reader proto, size_t batch, size_t timeslice,
               const std::vector<isaac::SharedBuffer>& buffers, State<K, N>& state) {
  const auto data_proto = proto.getData();
  const size_t data_size = data_proto.size();
  if (data_size > 0) {
    if (data_size != N) {
      return false;
    }
    for (size_t i = 0; i < data_size; i++) {
      state.elements[i] = data_proto[i];
    }
    return true;
  }

  const auto pack_proto = proto.getPack();

  // TODO Currently we only support double
  if (pack_proto.getElementType() != ::ElementType::FLOAT64) {
    return false;
  }

  const auto sizes_proto = pack_proto.getSizes();
  const size_t rank = sizes_proto.size();
  // TODO Currently we only support rank 3
  if (rank != 3) {
    return false;
  }

  TensorConstView3d tensor;
  if (!FromProto(pack_proto, buffers, tensor)) {
    return false;
  }

  if (batch < 0 || tensor.dimensions()[0] <= batch) {
    return false;
  }

  if (timeslice < 0 || tensor.dimensions()[1] <= timeslice) {
    return false;
  }

  for (size_t i = 0; i < N; i++) {
    state.elements[i] = tensor(batch, timeslice, i);
  }
  return true;
}

template <typename K, size_t N>
bool FromProto(StateProto::Reader proto, size_t timeslice,
               const std::vector<isaac::SharedBuffer>& buffers, State<K, N>& state) {
  return FromProto(proto, 0, timeslice, buffers, state);
}

template <typename K, size_t N>
bool FromProto(StateProto::Reader proto, const std::vector<isaac::SharedBuffer>& buffers,
               State<K, N>& state) {
  return FromProto(proto, 0, 0, buffers, state);
}

}  // namespace state
}  // namespace isaac
