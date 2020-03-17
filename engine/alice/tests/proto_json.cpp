/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "capnp/compat/json.h"
#include "capnp/dynamic.h"
#include "capnp/message.h"
#include "capnp/schema.h"

#include <memory>

#include "engine/alice/alice.hpp"
#include "engine/alice/message.hpp"
#include "engine/core/logger.hpp"
#include "engine/gems/serialization/json.hpp"
#include "gtest/gtest.h"
#include "messages/messages.hpp"
#include "messages/proto_registry.hpp"

namespace isaac {
namespace alice {

TEST(JsonToProto, BackForth) {
  const uint64_t proto_id = ::capnp::typeId<Vector2iProto>();
  // Proto data to start with
  std::unique_ptr<::capnp::MallocMessageBuilder> capnp_message_builder;
  capnp_message_builder.reset(new ::capnp::MallocMessageBuilder());

  ::capnp::DynamicStruct::Builder builder =
      *GetRootBuilderByTypeId(proto_id, *capnp_message_builder);
  builder.set("x", 100);
  builder.set("y", 200);

  kj::ArrayPtr<const kj::ArrayPtr<const ::capnp::word>> segments =
      capnp_message_builder->getSegmentsForOutput();

  // proto to json
  ::capnp::ReaderOptions options;
  options.traversalLimitInWords = kj::maxValue;
  ::capnp::SegmentArrayMessageReader reader(segments, options);

  ::capnp::JsonCodec codec;
  std::string json = (::kj::StringPtr)codec.encode(*GetRootReaderByTypeId(proto_id, reader));

  // json to proto
  ::capnp::MallocMessageBuilder decode_message;
  ::capnp::DynamicStruct::Builder root = *GetRootBuilderByTypeId(proto_id, decode_message);

  codec.decode((::kj::StringPtr)json.c_str(), root);

  // Checks data
  ::capnp::SegmentArrayMessageReader decode_reader(decode_message.getSegmentsForOutput(), options);
  ::capnp::DynamicStruct::Reader vector_reader = *GetRootReaderByTypeId(proto_id, decode_reader);
  EXPECT_EQ(vector_reader.get("x").as<int>(), 100);
  EXPECT_EQ(vector_reader.get("y").as<int>(), 200);
}

class ProtoMessageTestCodelet : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_proto1());
    synchronize(rx_proto1(), rx_proto2());
  }
  void tick() override {
    const ProtoMessageBase* proto1 = static_cast<const ProtoMessageBase*>(rx_proto1().getMessage());
    EXPECT_NE(proto1, nullptr);
    const ProtoMessageBase* proto2 = static_cast<const ProtoMessageBase*>(rx_proto2().getMessage());
    EXPECT_NE(proto2, nullptr);

    // Checks proto data match
    auto reader1 = rx_proto1().getProto();
    auto reader2 = rx_proto2().getProto();

    std::string frame1 = reader1.getPlanFrame();
    std::string frame2 = reader2.getPlanFrame();

    EXPECT_EQ(frame1, frame2);

    auto poses1 = reader1.getPoses();
    auto poses2 = reader2.getPoses();
    EXPECT_EQ(poses1.size(), poses2.size());
    EXPECT_GT(poses1.size(), 0);
    for (size_t i = 0; i < poses1.size(); ++i) {
      auto pose1 = poses1[i];
      auto pose2 = poses2[i];

      auto translation1 = pose1.getTranslation();
      auto translation2 = pose2.getTranslation();

      EXPECT_EQ(translation1.getX(), translation2.getX());
      EXPECT_EQ(translation1.getY(), translation2.getY());

      auto q1 = pose1.getRotation().getQ();
      auto q2 = pose2.getRotation().getQ();
      EXPECT_EQ(q1.getX(), q2.getX());
      EXPECT_EQ(q1.getY(), q2.getY());
    }
  }
  void stop() override { EXPECT_GT(getTickCount(), 0); }

  // RXs for proto message. Accepts all ProtoMessage though have Plan2Proto to play safe.
  ISAAC_PROTO_RX(Plan2Proto, proto1);
  ISAAC_PROTO_RX(Plan2Proto, proto2);
};

class JsonMessageTestCodelet : public Codelet {
 public:
  void start() override {
    tickOnMessage(rx_json1());
    synchronize(rx_json1(), rx_json2());
  }
  void tick() override {
    EXPECT_EQ(rx_json1().get().dump(), rx_json2().get().dump());
    EXPECT_EQ(rx_json1().acqtime(), rx_json2().acqtime());
  }
  void stop() override { EXPECT_GT(getTickCount(), 0); }

  // RXs for proto message. Accepts all ProtoMessage though have Plan2Proto to play safe.
  ISAAC_RAW_RX(nlohmann::json, json1);
  ISAAC_RAW_RX(nlohmann::json, json2);
};

TEST(JsonToProto, ProtoCheck) {
  Application app(serialization::LoadJsonFromFile("engine/alice/tests/proto_json.app.json"));

  app.startWaitStop(1.0);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::ProtoMessageTestCodelet);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::JsonMessageTestCodelet);
