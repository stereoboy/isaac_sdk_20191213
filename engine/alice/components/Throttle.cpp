/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "Throttle.hpp"

#include <string>
#include <vector>

#include "engine/alice/components/MessageLedger.hpp"
#include "engine/alice/node.hpp"

namespace isaac {
namespace alice {

void Throttle::initialize() {
  stopwatch_.setClock(node()->clock());
  ledger_ = node()->getComponent<MessageLedger>();
  ledger_->addOnConnectAsRxCallback(
      [=](const MessageLedger::Endpoint& tx, const MessageLedger::Endpoint& rx) {
        if (get_use_signal_channel() && rx.tag == get_signal_channel()) {
          ledger_->addOnMessageCallback(rx, tx.component,
              [=](ConstMessageBasePtr message) {
                onSignalMessage(message);
              });
          return;
        }
        if (rx.tag == get_data_channel()) {
          ledger_->addOnMessageCallback(rx, tx.component,
              [=](ConstMessageBasePtr message) {
                onDataMessage(message);
              });
          return;
        }
      });
}

void Throttle::onDataMessage(ConstMessageBasePtr signal_message) {
  // If the signal channel is enabled we will not proceed here, but use onSignalMessage instead.
  if (get_use_signal_channel()) return;
  // Publish the latest (new) message on the data channel
  ledger_->readLatestNew({this, get_data_channel()},
      [=](const ConstMessageBasePtr& message) { publish(message); });
}

void Throttle::onSignalMessage(ConstMessageBasePtr signal_message) {
  // If the signal channel is disabled we will not proceed here, but use onDataMessage instead.
  if (!get_use_signal_channel()) return;
  if (auto maybe_acqtime_tolerance = try_get_acqtime_tolerance()) {
    // Choose the first message on the data channel which lies within tolerance.
    const int64_t signal_acqtime = signal_message->acqtime;
    const int64_t acqtime_tolerance = *maybe_acqtime_tolerance;
    ledger_->checkChannelMessages({this, get_data_channel()},
        [=](const std::vector<ConstMessageBasePtr>& history) {
          for (const auto& message : history) {
            const int64_t delta = signal_acqtime - message->acqtime;
            if (std::abs(delta) < acqtime_tolerance) {
              publish(message);
              return;
            }
          }
        });
  } else {
    // Choose the latest message on the data channel.
    ledger_->checkChannelMessages({this, get_data_channel()},
        [=](const std::vector<ConstMessageBasePtr>& history) {
          if (!history.empty()) {
            publish(history.back());
          }
        });
  }
}

void Throttle::publish(ConstMessageBasePtr message) {
  // Publish the data, but not too fast
  if (stopwatch_.interval(get_minimum_interval())) {
    ledger_->provide({this, get_output_channel()}, message);
  }
}

}  // namespace alice
}  // namespace isaac
