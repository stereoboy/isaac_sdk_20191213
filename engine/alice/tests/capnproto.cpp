/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <iostream>

#include "capnp/message.h"
#include "capnp/serialize-packed.h"
#include "engine/alice/tests/bar.capnp.h"
#include "engine/alice/tests/foo.capnp.h"
#include "gtest/gtest.h"

TEST(Alice, CapnProto) {
{
  ::capnp::MallocMessageBuilder message;
  Foo::Builder foo = message.initRoot<Foo>();
  foo.setNumber(42);
  foo.setAmount(3.1415);

  std::cout << "================================================================" << std::endl;
  writePackedMessageToFd(1, message);
  std::cout << std::endl;
  std::cout << "================================================================" << std::endl;
}

{
  ::capnp::MallocMessageBuilder message;
  Bar::Builder bar = message.initRoot<Bar>();
  bar.setText("hello world");

  std::cout << "================================================================" << std::endl;
  writePackedMessageToFd(1, message);
  std::cout << std::endl;
  std::cout << "================================================================" << std::endl;
}
}
