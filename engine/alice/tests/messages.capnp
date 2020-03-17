#####################################################################################
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#####################################################################################
@0xdc430ce3220051e8;

struct IntProto {
  value @0: Int32;
}

struct FooProto {
  count @0: Int32;
  value @1: Float32;
  text @2: Text;
}

struct BarProto {
  counter @0: Int32;
  hop @1: Int32;
}

struct BulkyProto {
  foo @0: Int32;
  bar @1: Text;
  data1 @2: Data;
  data2 @3: Data;
  data3 @4: Data;
  data4 @5: Data;
}
