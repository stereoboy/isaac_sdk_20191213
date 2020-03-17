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
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/node_backend.hpp"
#include "engine/alice/tests/foo_transmission.hpp"
#include "engine/core/logger.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

class MyCodelet1 : public Codelet {
 private:
  ISAAC_PARAM(int, foo, 42)
};

class MyCodelet2 : public Codelet {
 private:
  ISAAC_PARAM(int, bar, 27)
};

TEST(Alice, LoadNodes) {
  const char app_json_text[] =
R"???({
  "name": "AliceLoadNodes",
  "config": {
    "test": {"codelet 2": {"bar": 31}},
    "bar": {"codelet 2": {"bar": 33}}
  },
  "graph": {
    "nodes": [
      {
        "name": "test",
        "components": [
          {"name": "codelet 1", "type": "isaac::alice::MyCodelet1"},
          {"name": "codelet 2", "type": "isaac::alice::MyCodelet2"},
          {"name": "codelet 1 (b)", "type": "isaac::alice::MyCodelet1"}
        ]
      },
      {
        "name": "bar",
        "components": [
          { "name": "codelet 2", "type": "isaac::alice::MyCodelet2" }
        ]
      }
    ],
    "edges": []
  }
})???";

  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node1 = app.findNodeByName("test");
  ASSERT_NE(node1, nullptr);
  EXPECT_EQ(node1->name(), "test");
  auto comps1 = node1->getComponents<MyCodelet1>();
  ASSERT_EQ(comps1.size(), 2);
  auto comps2 = node1->getComponents<MyCodelet2>();
  ASSERT_EQ(comps2.size(), 1);
  EXPECT_EQ(comps2.front()->get_bar(), 31);
  Node* node2 = app.findNodeByName("bar");
  ASSERT_NE(node2, nullptr);
  EXPECT_EQ(node2->name(), "bar");
  auto comp3 = node2->getComponent<MyCodelet2>();
  EXPECT_EQ(comp3->get_bar(), 33);
}

TEST(Alice, DISABLED_LoadNodesDeath) {
  ApplicationJsonLoader loader;

  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "nAmez": "test",
      "components": [
        {"name": "codelet 1", "type": "isaac::alice::MyCodelet1"},
        {"name": "codelet 1 (b)", "type": "isaac::alice::MyCodelet1"}
      ]
    }
  ],
  "edges": []
})???"), ".?");

  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "test",
      "coMpONentZ": [
        {"name": "codelet 1", "type": "isaac::alice::MyCodelet1"},
        {"name": "codelet 1 (b)", "type": "isaac::alice::MyCodelet1"}
      ]
    }
  ],
  "edges": []
})???"), ".?");

  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "test",
      "components": [
        {"nAme_": "codelet 1", "type": "isaac::alice::MyCodelet1"},
        {"name": "codelet 1 (b)", "type": "isaac::alice::MyCodelet1"}
      ]
    }
  ],
  "edges": []
})???"), ".?");

  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "test",
      "components": [
        {"name": "codelet 1", "type": "isaac::alice::MyCodelet1"},
        {"name": "codelet 1 (b)", "typo": "isaac::alice::MyCodelet1"}
      ]
    }
  ],
  "edges": []
})???"), ".?");
}

TEST(Alice, LoadEmptyEdges) {
  ApplicationJsonLoader loader;

  loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "test_node",
      "components": [
        {"name": "test_component", "type": "isaac::alice::MyCodelet1"}
      ]
    }
  ]
})???");

  Application app(loader);
  Node* node1 = app.findNodeByName("test_node");
  ASSERT_NE(node1, nullptr);
}

TEST(Alice, SaveLoadNodes) {
  // create app
  std::string json_str;
  {
    Application app;
    Node* node1 = app.createNode("test");
    EXPECT_EQ(node1->name(), "test");
    node1->addComponent<MyCodelet1>();
    node1->addComponent<MyCodelet2>();
    node1->addComponent<MyCodelet1>("second");
    Node* node2 = app.createNode("bar");
    EXPECT_EQ(node2->name(), "bar");
    node2->addComponent<MyCodelet2>();
    app.startWaitStop(0.10);  // FIXME
    json_str = ApplicationJsonLoader::GetGraphJson(app).dump(2);
  }
  LOG_INFO("%s", json_str.c_str());
  // load app
  {
    ApplicationJsonLoader loader;
    loader.loadGraphFromText(json_str);
    Application app(loader);
    EXPECT_EQ(app.backend()->node_backend()->numNodes(), 3);
    Node* node1 = app.findNodeByName("test");
    ASSERT_NE(node1, nullptr);
    EXPECT_EQ(node1->name(), "test");
    auto comps1 = node1->getComponents<MyCodelet1>();
    ASSERT_EQ(comps1.size(), 2);
    auto comps2 = node1->getComponents<MyCodelet2>();
    ASSERT_EQ(comps2.size(), 1);
    EXPECT_EQ(comps2.front()->get_bar(), 27);
    Node* node2 = app.findNodeByName("bar");
    ASSERT_NE(node2, nullptr);
    EXPECT_EQ(node2->name(), "bar");
    auto comp3 = node1->getComponent<MyCodelet2>();
    ASSERT_NE(comp3, nullptr);
    EXPECT_EQ(comp3->get_bar(), 27);
  }
}

TEST(Alice, LoadNodesAndEdges) {
  const char app_json_text[] =
R"???({
  "name": "AliceLoadNodesAndEdges",
  "config": {
    "alice": {
      "transmitter": {
        "expected_tick_count": 6,
        "expected_tick_count_tolerance": 1,
        "tick_period": "100ms"
      }
    },
    "bob": {
      "receiver": {
        "expected_tick_count": 6,
        "expected_tick_count_tolerance": 1
      }
    }
  },
  "graph": {
    "nodes": [
      {
        "name": "alice",
        "components": [
          {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
          {"name": "transmitter", "type": "isaac::alice::FooTransmitter"}
        ]
      },
      {
        "name": "bob",
        "components": [
          {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
          {"name": "receiver", "type": "isaac::alice::FooReceiver"}
        ]
      }
    ],
    "edges": [
      { "source": "alice/transmitter/foo", "target": "bob/receiver/foo" }
    ]
  }
})???";

  Application app(serialization::LoadJsonFromText(app_json_text));

  Node* node1 = app.findNodeByName("alice");
  ASSERT_NE(node1, nullptr);
  ASSERT_NE(node1->findComponentByName("transmitter"), nullptr);
  Node* node2 = app.findNodeByName("bob");
  ASSERT_NE(node2, nullptr);
  ASSERT_NE(node2->findComponentByName("receiver"), nullptr);

  app.startWaitStop(0.55);
}

TEST(Alice, LoadEmptyNodesDeath) {
  ApplicationJsonLoader loader;
  EXPECT_DEATH(loader.loadGraphFromText("{}"), "");
}

TEST(Alice, MissingSourceDeath) {
  ApplicationJsonLoader loader;
  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "alice",
      "components": [
        {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
        {"name": "transmitter", "type": "isaac::alice::FooTransmitter"}
      ]
    },
    {
      "name": "bob",
      "components": [
        {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
        {"name": "receiver", "type": "isaac::alice::FooReceiver"}
      ]
    }
  ],
  "edges": [
    { "ZourZe": "alice/transmitter/foo", "target": "bob/receiver/foo" }
  ]
})???"), "");
}

TEST(Alice, MissingTargetDeath) {
  ApplicationJsonLoader loader;
  EXPECT_DEATH(loader.loadGraphFromText(R"???({
  "nodes": [
    {
      "name": "alice",
      "components": [
        {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
        {"name": "transmitter", "type": "isaac::alice::FooTransmitter"}
      ]
    },
    {
      "name": "bob",
      "components": [
        {"name": "message_ledger", "type": "isaac::alice::MessageLedger"},
        {"name": "receiver", "type": "isaac::alice::FooReceiver"}
      ]
    }
  ],
  "edges": [
    { "source": "alice/transmitter/foo", "tArGeT": "bob/receiver/foo" }
  ]
})???"), "");
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet1);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::MyCodelet2);
