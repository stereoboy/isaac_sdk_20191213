/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cstdlib>

#include "engine/core/logger.hpp"

// Prints a panic message and aborts the program
#define PANIC(...) \
  { \
    ::isaac::logger::Log(__FILE__, __LINE__, ::isaac::logger::Severity::PANIC, __VA_ARGS__); \
    std::abort(); \
  }

// Checks if an expression evaluates to true. If not prints a panic message and aborts the program.
#define ASSERT(expr, ...) \
  if (!(expr)) { \
    ::isaac::logger::Log(__FILE__, __LINE__, ::isaac::logger::Severity::PANIC, __VA_ARGS__); \
    std::abort(); \
  }
