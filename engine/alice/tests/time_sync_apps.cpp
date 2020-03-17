/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include <thread>

#include "engine/alice/alice.hpp"
#include "engine/alice/tests/messages.hpp"
#include "gtest/gtest.h"

namespace isaac {
namespace alice {

namespace {

// Ports to be used for tcp communucation
constexpr int kPort1 = 42005;
constexpr int kPort2 = 42006;

// Name of the for the edges
constexpr char kTag[] = "tag_name";

// Value transmitted with messages
constexpr int kValue = 42;

// Time values to be used
constexpr double kOffsetReceiver = 3.0;
constexpr double kOffsetSecondPub = 1.5;
constexpr double kTimeTotal = 5.0;
constexpr double kThreshold = 0.2;

}  // namespace

// Transmits a dummy message repeatedly
class PingTx : public Codelet {
 public:
  void start() override {
    set_tick_period("100ms");
    tickPeriodically();
  }
  void tick() override {
    auto outmsg = tx_out().initProto();
    outmsg.setValue(kValue);
    tx_out().publish();
  }
  ISAAC_PROTO_TX(IntProto, out);
};

// Saves the acqtime of the first message received
class PingRx : public Codelet {
 public:
  void start() override {
    maybe_acqtime_seconds_ = std::nullopt;
    tickOnMessage(rx_in());
  }
  void tick() override {
    auto msg = rx_in().getProto();
    EXPECT_EQ(kValue, msg.getValue());
    maybe_acqtime_seconds_ = ToSeconds(rx_in().acqtime());
    reportSuccess();  // tick once
  }
  ISAAC_PROTO_RX(IntProto, in);
  double acqtime_seconds() const {
    ASSERT(maybe_acqtime_seconds_, "PingRx has not ticked!");
    return *maybe_acqtime_seconds_;
  }

 private:
  std::optional<double> maybe_acqtime_seconds_;
};

void Test1Receiver1Publisher(bool time_sync) {
  std::thread t1([time_sync] {
    Application appPublisher;
    // tcp node
    Node* tcp_pub_node = appPublisher.createMessageNode("tcp_pub");
    auto* tcp_pub_comp = tcp_pub_node->addComponent<TcpPublisher>();
    tcp_pub_comp->async_set_port(kPort1);
    if (time_sync) {
      // Add time-sync to tcp node
      tcp_pub_node->addComponent<TimeSynchronizer>();
    }
    // message generator
    Node* generator_node = appPublisher.createMessageNode("generator");
    auto* generator_comp = generator_node->addComponent<PingTx>();
    // edge
    Connect(generator_comp->tx_out(), tcp_pub_comp, kTag);
    appPublisher.startWaitStop(kTimeTotal);
  });

  std::thread t2([time_sync] {
    Application appSubscriber;
    // tcp node
    Node* tcp_sub_node = appSubscriber.createMessageNode("tcp_sub");
    auto* tcp_sub_comp = tcp_sub_node->addComponent<TcpSubscriber>();
    tcp_sub_comp->async_set_port(kPort1);
    tcp_sub_comp->async_set_host("127.0.0.1");
    tcp_sub_comp->async_set_reconnect_interval(0.1);
    if (time_sync) {
      // Add time-sync to tcp node
      tcp_sub_node->addComponent<TimeSynchronizer>();
    }
    // message reader
    Node* reader_node = appSubscriber.createMessageNode("reader");
    auto* reader_comp = reader_node->addComponent<PingRx>();
    // edge
    Connect(tcp_sub_comp, kTag, reader_comp->rx_in());
    Sleep(SecondsToNano(kOffsetReceiver));
    appSubscriber.startWaitStop(kTimeTotal - kOffsetReceiver);
    const double acqtime_seconds = reader_comp->acqtime_seconds();
    LOG_INFO("Received with acqtime: %f", acqtime_seconds);
    if (!time_sync) {
      // Without time synchronization, first acqtime is large
      EXPECT_NEAR(acqtime_seconds, kOffsetReceiver, kThreshold);
    } else {
      // With time synchronization, first acqtime should be small
      EXPECT_GT(acqtime_seconds, 0.0);
      EXPECT_LE(acqtime_seconds, kThreshold);
    }
  });

  t2.join();
  t1.join();
}

void Test1Receiver2Publisher(bool time_sync) {
  std::thread t1a([time_sync] {
    Application appPublisher;
    // tcp node
    Node* tcp_pub_node = appPublisher.createMessageNode("tcp_pub");
    auto* tcp_pub_comp = tcp_pub_node->addComponent<TcpPublisher>();
    tcp_pub_comp->async_set_port(kPort1);
    if (time_sync) {
      // Add time-sync to tcp node
      tcp_pub_node->addComponent<TimeSynchronizer>();
    }
    // message generator
    Node* generator_node = appPublisher.createMessageNode("generator");
    auto* generator_comp = generator_node->addComponent<PingTx>();
    // edge
    Connect(generator_comp->tx_out(), tcp_pub_comp, kTag);
    appPublisher.startWaitStop(kTimeTotal);
  });

  std::thread t1b([time_sync] {
    Application appPublisher;
    // tcp node
    Node* tcp_pub_node = appPublisher.createMessageNode("tcp_pub");
    auto* tcp_pub_comp = tcp_pub_node->addComponent<TcpPublisher>();
    tcp_pub_comp->async_set_port(kPort2);
    if (time_sync) {
      // Add time-sync to tcp node
      tcp_pub_node->addComponent<TimeSynchronizer>();
    }
    // message generator
    Node* generator_node = appPublisher.createMessageNode("generator");
    auto* generator_comp = generator_node->addComponent<PingTx>();
    // edge
    Connect(generator_comp->tx_out(), tcp_pub_comp, kTag);
    Sleep(SecondsToNano(kOffsetSecondPub));
    appPublisher.startWaitStop(kTimeTotal - kOffsetSecondPub);
  });

  std::thread t2([time_sync] {
    Application appSubscriber;
    // tcp node
    Node* tcp_sub_node = appSubscriber.createMessageNode("tcp_sub");
    auto* tcp_sub_comp1 = tcp_sub_node->addComponent<TcpSubscriber>("1");
    auto* tcp_sub_comp2 = tcp_sub_node->addComponent<TcpSubscriber>("2");
    tcp_sub_comp1->async_set_port(kPort1);
    tcp_sub_comp2->async_set_port(kPort2);
    tcp_sub_comp1->async_set_host("127.0.0.1");
    tcp_sub_comp2->async_set_host("127.0.0.1");
    tcp_sub_comp1->async_set_reconnect_interval(0.1);
    tcp_sub_comp2->async_set_reconnect_interval(0.1);
    if (time_sync) {
      // Add time-sync to tcp node
      tcp_sub_node->addComponent<TimeSynchronizer>();
    }
    // message reader
    Node* reader_node = appSubscriber.createMessageNode("reader");
    auto* reader_comp1 = reader_node->addComponent<PingRx>("1");
    auto* reader_comp2 = reader_node->addComponent<PingRx>("2");
    // edge
    Connect(tcp_sub_comp1, kTag, reader_comp1->rx_in());
    Connect(tcp_sub_comp2, kTag, reader_comp2->rx_in());
    Sleep(SecondsToNano(kOffsetReceiver));
    appSubscriber.startWaitStop(kTimeTotal - kOffsetReceiver);
    const double acqtime_seconds1 = reader_comp1->acqtime_seconds();
    const double acqtime_seconds2 = reader_comp2->acqtime_seconds();
    LOG_INFO("Received with acqtime1: %f", acqtime_seconds1);
    LOG_INFO("Received with acqtime2: %f", acqtime_seconds2);
    if (!time_sync) {
      // Without time synchronization, first acqtime is large
      EXPECT_NEAR(acqtime_seconds1, kOffsetReceiver, kThreshold);
      ASSERT(kOffsetReceiver > kOffsetSecondPub, "mistake in test logic");
      EXPECT_NEAR(acqtime_seconds2, kOffsetReceiver - kOffsetSecondPub, kThreshold);
    } else {
      // With time synchronization, first acqtime should be small
      EXPECT_NEAR(acqtime_seconds1, acqtime_seconds2, kThreshold);
    }

    //    EXPECT_GT(acqtime_seconds, 8.0 - 1.0);
    //  EXPECT_LT(acqtime_seconds, 8.0 + 1.0);
  });

  t2.join();
  t1a.join();
  t1b.join();
}

TEST(TimeSyncApps, WithoutTimeSync) {
  const bool time_sync = false;
  Test1Receiver1Publisher(time_sync);
}

TEST(TimeSyncApps, WithTimeSync) {
  const bool time_sync = true;
  Test1Receiver1Publisher(time_sync);
}

TEST(TimeSyncApps, WithoutTimeSync2Pubs) {
  const bool time_sync = false;
  Test1Receiver2Publisher(time_sync);
}

TEST(TimeSyncApps, WithTimeSync2Pubs) {
  const bool time_sync = true;
  Test1Receiver2Publisher(time_sync);
}

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingRx);
ISAAC_ALICE_REGISTER_CODELET(isaac::alice::PingTx);
