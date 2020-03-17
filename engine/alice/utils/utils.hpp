/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include "engine/alice/message.hpp"
#include "engine/gems/cask/cask.hpp"

namespace isaac {
namespace alice {

// Writes message to cask using message uuid as key
void WriteMessageToCask(ConstMessageBasePtr message, cask::Cask& cask);

// Reads message from cask with uuid
MessageBasePtr ReadMessageFromCask(const Uuid& message_uuid, cask::Cask& cask);

}  // namespace alice
}  // namespace isaac
