/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/core/string_literal.hpp"

#include "gtest/gtest.h"

namespace isaac {

TEST(StringLiteral, TestHello) {
  auto x = string_literal("hello");
  EXPECT_EQ(x.size(), 5);
  ASSERT_STREQ((char const*)x, "hello");
  EXPECT_EQ(x[0], 'h');
  EXPECT_EQ(x[1], 'e');
  EXPECT_EQ(x[2], 'l');
  EXPECT_EQ(x[3], 'l');
  EXPECT_EQ(x[4], 'o');
  EXPECT_EQ(x[5], '\0');
  EXPECT_EQ(x[6], '\0');
  EXPECT_EQ(x[7], '\0');
}

TEST(StringLiteral, TestEmpty) {
  auto x = string_literal("");
  EXPECT_EQ(x.size(), 0);
  ASSERT_STREQ((char const*)x, "");
  EXPECT_EQ(x[0], '\0');
  EXPECT_EQ(x[1], '\0');
  EXPECT_EQ(x[2], '\0');
}

TEST(StringLiteral, TestWithNull) {
  auto x = string_literal("abc\0d\0");
  EXPECT_EQ(x.size(), 6);
  ASSERT_STREQ((char const*)x, "abc\0d\0");
  EXPECT_EQ(x[0], 'a');
  EXPECT_EQ(x[1], 'b');
  EXPECT_EQ(x[2], 'c');
  EXPECT_EQ(x[3], '\0');
  EXPECT_EQ(x[4], 'd');
  EXPECT_EQ(x[5], '\0');
  EXPECT_EQ(x[6], '\0');
  EXPECT_EQ(x[7], '\0');
}

TEST(StringLiteral, TestFromConstexpr) {
  constexpr char str[] = "xyz";
  auto x = string_literal(str);
  EXPECT_EQ(x.size(), 3);
  ASSERT_STREQ((char const*)x, "xyz");
  ASSERT_STREQ((char const*)x, str);
}

}  // namespace isaac
