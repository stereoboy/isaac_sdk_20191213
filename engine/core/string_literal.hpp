/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cstddef>

namespace isaac {

// A convenient type to handle compile-time string literals
class string_literal {
 public:
  // Creates a string literal based on an existing literal.
  template <size_t N>
  constexpr string_literal(const char (&a)[N])
  : characters_(a), size_(N-1) {}

  // The length of the string literal without null-terminator.
  constexpr size_t size() const { return size_; }

  // Converts automatically to a pointer to characters.
  constexpr operator char const*() const { return characters_; }

  // Accesses the character at position `index` or returns null-terminator if out of bounds.
  constexpr char operator[](size_t index) {
    return index < size_ ? characters_[index] : '\0';
  }

 private:
  const char* const characters_;
  const size_t size_;
};

}  // namespace isaac
