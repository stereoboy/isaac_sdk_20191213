/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "engine/alice/alice.hpp"
#include "engine/alice/backend/application_json_loader.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyConfigTest : public Codelet {
 public:
  ISAAC_PARAM(std::string, foo, "yeah!");
  ISAAC_PARAM(int, bar, 42);
  ISAAC_PARAM(double, pie, 3.1415);
  ISAAC_PARAM(bool, xin, true);
  ISAAC_PARAM(nlohmann::json, some, {});
  ISAAC_PARAM(std::vector<std::string>, list_of_strings, {});
};

TEST(Alice, Config) {
  Application app;
  Node* node = app.createNode("test");
  auto* codelet = node->addComponent<MyConfigTest>();

  EXPECT_STREQ(codelet->get_foo().c_str(), "yeah!");
  EXPECT_EQ(codelet->get_bar(), 42);
  EXPECT_EQ(codelet->get_pie(), 3.1415);
  EXPECT_TRUE(codelet->get_xin());

  auto list_of_strings = codelet->get_list_of_strings();
  EXPECT_TRUE(list_of_strings.empty());

  app.startWaitStop(0.10);

  auto list_of_strings2 = codelet->get_list_of_strings();
  EXPECT_TRUE(list_of_strings2.empty());
}

TEST(Config, LoadListOfStrings) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigLoadListOfStrings",
  "config": {
    "test node": {
      "test component": {
        "bar": 37,
        "list_of_strings": ["a", "bb", "123"]
      }
    }
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  EXPECT_EQ(codelet->get_bar(), 37);

  auto list_of_strings = codelet->get_list_of_strings();
  ASSERT_EQ(list_of_strings.size(), 3);
  ASSERT_STREQ(list_of_strings[0].c_str(), "a");
  ASSERT_STREQ(list_of_strings[1].c_str(), "bb");
  ASSERT_STREQ(list_of_strings[2].c_str(), "123");

  app.startWaitStop(0.10);
}

TEST(Config, LoadJsonJson) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigLoadJsonJson",
  "config": {
    "test node": {
      "test component": {
        "bar": 37,
        "some": {"cake": 3.7, "values": [7,-2,3]}
      }
    }
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  EXPECT_EQ(codelet->get_bar(), 37);
  auto json = codelet->get_some();
  ASSERT_NE(json.find("cake"), json.end());
  EXPECT_EQ(json["cake"], 3.7);
  ASSERT_NE(json.find("values"), json.end());
  ASSERT_TRUE(json["values"].is_array());
  ASSERT_EQ(json["values"].size(), 3);
  EXPECT_EQ(json["values"][0], 7);
  EXPECT_EQ(json["values"][1], -2);
  EXPECT_EQ(json["values"][2], 3);

  app.startWaitStop(0.10);
}

TEST(Config, Load) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigLoad",
  "config": {
    "test node": {
      "test component": {
        "foo": "hola",
        "bar": 37,
        "pie": 1.4153,
        "xin": false
      }
    }
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  EXPECT_STREQ(codelet->get_foo().c_str(), "hola");
  EXPECT_EQ(codelet->get_bar(), 37);
  EXPECT_EQ(codelet->get_pie(), 1.4153);
  EXPECT_FALSE(codelet->get_xin());

  app.startWaitStop(0.10);
}

TEST(Config, LoadIntAsDouble) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigLoadIntAsDouble",
  "config": {
    "test node": {
      "test component": {
        "pie": 21
      }
    }
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  EXPECT_STREQ(codelet->get_foo().c_str(), "yeah!");
  EXPECT_EQ(codelet->get_bar(), 42);
  EXPECT_EQ(codelet->get_pie(), 21.0);
  EXPECT_TRUE(codelet->get_xin());

  app.startWaitStop(0.10);
}

TEST(Config, LoadMultiple) {
  ApplicationJsonLoader loader;

  loader.loadConfigFromText(R"???(
{
  "test node": {
    "test component": {
      "bar": 37,
      "pie": 1.4153,
      "foo": "not really"
    }
  }
})???");

  loader.loadConfigFromText(R"???(
{
  "test node": {
    "test component": {
      "foo": "hola",
      "xin": false
    }
  }
})???");

  Application app(loader);

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  EXPECT_STREQ(codelet->get_foo().c_str(), "hola");
  EXPECT_EQ(codelet->get_bar(), 37);
  EXPECT_EQ(codelet->get_pie(), 1.4153);
  EXPECT_FALSE(codelet->get_xin());

  app.startWaitStop(0.10);
}

TEST(Config, LoadInvalid) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigLoadInvalid",
  "config": {
    "test node": {
      "test component": {
        "foo": "hola",
        "bar: 37,
        "pie": 1.4153,
      }
    }
  }
})???";
  EXPECT_DEATH(Application(serialization::LoadJsonFromText(app_json_text)), ".?parsing.?");
}

TEST(Config, GetAllJson) {
  constexpr char app_json_text[] =
R"???({
  "name": "ConfigGetAllJson",
  "config": {
    "test node": {
      "test component": {
        "foo": "hola",
        "bar": 37,
        "pie": 1.4153,
        "xin": false
      }
    }
  }
})???";
  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node = app.createNode("test node");
  auto* codelet = node->addComponent<MyConfigTest>("test component");

  auto json = codelet->node()->config().getAll(codelet);
  EXPECT_EQ(json["foo"], "hola");
  EXPECT_EQ(json["bar"], 37);
  EXPECT_EQ(json["pie"], 1.4153);
  EXPECT_EQ(json["xin"], false);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyConfigTest);
