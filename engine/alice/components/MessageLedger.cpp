/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "MessageLedger.hpp"

#include <algorithm>
#include <memory>
#include <shared_mutex>  // NOLINT
#include <string>
#include <utility>
#include <vector>

#include "engine/alice/application.hpp"
#include "engine/alice/backend/backend.hpp"
#include "engine/alice/backend/message_ledger_backend.hpp"
#include "engine/alice/node.hpp"
#include "engine/gems/scheduler/scheduler.hpp"

namespace isaac {
namespace alice {
namespace {
// Copies a text into a string and return the pointer of the next character to be written.
// If append_slash is true, it will add / at the end
char* AppendToString(const std::string& text, bool append_slash, char* ptr) {
  std::copy(text.begin(), text.end(), ptr);
  ptr += text.size();
  if (append_slash) (*ptr++) = '/';
  return ptr;
}
}  // namespace

void MessageLedger::initialize() {
  // Notify the scheduler about all received messages to trigger codelets
  addOnConnectAsRxCallback(
    [this](const MessageLedger::Endpoint& tx, const MessageLedger::Endpoint& rx) {
      this->addOnMessageCallback(rx, tx.component,
          [this, rx, tx](ConstMessageBasePtr message) {
            this->notifyScheduler(rx, message->pubtime);
          });
    });
}

void MessageLedger::start() {
  reportSuccess();  // do not participate in status updates TODO solver differently
}

void MessageLedger::deinitialize() {
  disconnect();
}

std::string MessageLedger::Endpoint::name() const {
  std::string str;
  const std::string& node_name = component->node()->name();
  const std::string& comp_name = component->name();
  // Preallocate the memory and perform the copy directly in place. (needs +2 for the /)
  str.resize(2 + tag.size() + comp_name.size() + node_name.size());
  char* ptr = AppendToString(node_name, true, &str[0]);
  ptr = AppendToString(comp_name, true, ptr);
  ptr = AppendToString(tag, false, ptr);
  return str;
}

std::string MessageLedger::Endpoint::nameWithApp() const {
  std::string str;
  const std::string& app_name = component->node()->app()->name();
  const std::string& node_name = component->node()->name();
  const std::string& comp_name = component->name();
  // Preallocate the memory and perform the copy directly in place. (needs +3 for the /)
  str.resize(3 + tag.size() + comp_name.size() + node_name.size() + app_name.size());
  char* ptr = AppendToString(app_name, true, &str[0]);
  ptr = AppendToString(node_name, true, ptr);
  ptr = AppendToString(comp_name, true, ptr);
  ptr = AppendToString(tag, false, ptr);
  return str;
}

void MessageLedger::addOnConnectAsTxCallback(OnConnectCallback callback) {
  on_connect_as_tx_callbacks_.emplace_back(std::move(callback));
}

void MessageLedger::addOnConnectAsRxCallback(OnConnectCallback callback) {
  on_connect_as_rx_callbacks_.emplace_back(std::move(callback));
}

void MessageLedger::provide(const Endpoint& endpoint, ConstMessageBasePtr message) {
  getOrCreateChannel(endpoint)->addMessage(message, get_history());
  // call general callbacks
  std::lock_guard<std::mutex> lock(callbacks_mutex_);
  for (auto& callback : callbacks_) {
    callback(endpoint, message);
  }
}

void MessageLedger::notifyScheduler(const Endpoint& endpoint, int64_t target_time) const {
  node()->app()->backend()->scheduler()->notify(endpoint.name(), target_time);
}

void MessageLedger::addOnMessageCallback(const Endpoint& endpoint, const Component* source,
                                         OnMessageCallback callback) {
  getOrCreateChannel(endpoint)->addCallback(source, std::move(callback));
}

void MessageLedger::addOnMessageCallback(OnChannelMessageCallback callback) {
  std::lock_guard<std::mutex> lock(callbacks_mutex_);
  callbacks_.push_back(callback);
}

void MessageLedger::removeCustomer(const Component* source) {
  std::unique_lock<std::shared_timed_mutex> lock(channels_mutex_);
  for (auto& kvp : channels_) {
    kvp.second->removeSource(source);
  }
}

void MessageLedger::addSource(MessageLedger* source, const Component* customer) {
  std::unique_lock<std::mutex> lock(sources_mutex_);
  sources_.insert(std::make_pair(source, customer));
}

void MessageLedger::disconnect() {
  std::unique_lock<std::mutex> lock(sources_mutex_);
  for (auto& source : sources_) {
    source.first->removeCustomer(source.second);
  }
  sources_.clear();
}

void MessageLedger::readAllNew(
    std::function<void(const Endpoint&, const ConstMessageBasePtr&)> callback) const {
  std::shared_lock<std::shared_timed_mutex> lock(channels_mutex_);
  for (auto& kvp : channels_) {
    kvp.second->readAllNew(
        [&](const ConstMessageBasePtr& message) { callback(kvp.first, message); });
  }
}

void MessageLedger::readAllNew(const Endpoint& endpoint,
    std::function<void(const ConstMessageBasePtr&)> callback) const {
  if (auto channel = findChannel(endpoint)) {
    channel->readAllNew(std::move(callback));
  }
}

void MessageLedger::peekAllNew(const Endpoint& endpoint,
    std::function<void(const ConstMessageBasePtr&)> callback) const {
  if (auto channel = findChannel(endpoint)) {
    channel->peekAllNew(std::move(callback));
  }
}

void MessageLedger::readAllLatest(
    std::function<void(const Endpoint&, const ConstMessageBasePtr&)> callback) {
  std::shared_lock<std::shared_timed_mutex> lock1(channels_mutex_);
  for (auto& kvp : channels_) {
    kvp.second->checkMessages([&](const std::vector<ConstMessageBasePtr>& history) {
      if (!history.empty()) {
        callback(kvp.first, history.back());
      }
    });
  }
}

void MessageLedger::readLatestNew(const Endpoint& endpoint,
    std::function<void(const ConstMessageBasePtr&)> callback) const {
  ConstMessageBasePtr message;
  readAllNew(endpoint, [&](const ConstMessageBasePtr& ptr) { message = ptr; });
  if (message) callback(message);
}

void MessageLedger::peekLatestNew(const Endpoint& endpoint,
    std::function<void(const ConstMessageBasePtr&)> callback) const {
  ConstMessageBasePtr message;
  peekAllNew(endpoint, [&](const ConstMessageBasePtr& ptr) { message = ptr; });
  if (message) callback(message);
}

void MessageLedger::checkChannelMessages(const Endpoint& endpoint,
                                         OnMessageListCallback callback) const {
  if (auto ptr = findChannel(endpoint)) {
    ptr->checkMessages(std::move(callback));
  }
}

size_t MessageLedger::numSourceChannels() const {
  return sources_.size();
}

void MessageLedger::Channel::addCallback(const Component* source, OnMessageCallback callback) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  callbacks_[source].emplace_back(std::move(callback));
}

void MessageLedger::Channel::removeSource(const Component* source) {
  std::lock_guard<std::mutex> lock(channels_mutex_);
  callbacks_.erase(source);
}

void MessageLedger::Channel::addMessage(ConstMessageBasePtr message, int max) {
  // Append message
  {
    std::lock_guard<std::mutex> lock(messages_mutex_);
    messages_.push_back(message);
    // Limit history size
    const size_t max_size = static_cast<size_t>(max);
    if (messages_.size() > max_size) {
      const size_t num_removed = messages_.size() - max_size;
      unread_index_ = (unread_index_ < num_removed) ? 0 : (unread_index_ - num_removed);
      messages_.erase(messages_.begin(), std::next(messages_.begin(), num_removed));
    }
  }
  // Calls callbacks
  {
    std::lock_guard<std::mutex> lock(channels_mutex_);
    for (auto& kvp : callbacks_) {
      for (auto& callback : kvp.second) {
        callback(message);
      }
    }
  }
}

void MessageLedger::Channel::readAllNew(
    std::function<void(const ConstMessageBasePtr&)> callback) {
  std::lock_guard<std::mutex> lock(messages_mutex_);
  for (size_t i = unread_index_; i < messages_.size(); i++) {
    callback(messages_[i]);
  }
  unread_index_ = messages_.size();
}

void MessageLedger::Channel::peekAllNew(
    std::function<void(const ConstMessageBasePtr&)> callback) {
  std::lock_guard<std::mutex> lock(messages_mutex_);
  for (size_t i = unread_index_; i < messages_.size(); i++) {
    callback(messages_[i]);
  }
}

void MessageLedger::Channel::checkMessages(OnMessageListCallback callback) const {
  std::lock_guard<std::mutex> lock(messages_mutex_);
  callback(messages_);
}

std::shared_ptr<MessageLedger::Channel> MessageLedger::findChannel(const Endpoint& endpoint) const {
  std::shared_lock<std::shared_timed_mutex> lock(channels_mutex_);
  auto it = channels_.find(endpoint);
  if (it == channels_.end()) {
    return nullptr;
  } else {
    return it->second;
  }
}

std::shared_ptr<MessageLedger::Channel> MessageLedger::getOrCreateChannel(
      const Endpoint& endpoint) {
  // Try to find the channel under a shared lock.
  channels_mutex_.lock_shared();
  auto it = channels_.find(endpoint);
  if (it != channels_.end()) {
    std::shared_ptr<Channel> sptr = it->second;
    channels_mutex_.unlock_shared();
    return sptr;
  } else {
    // We need to create the channel first. We have to release the shared lock and acquire a
    // unique lock.
    channels_mutex_.unlock_shared();
    channels_mutex_.lock();
    it = channels_.emplace(std::make_pair(endpoint, std::make_shared<Channel>())).first;
    channels_mutex_.unlock();
    return it->second;
  }
}

void MessageLedger::connectAsTx(const Endpoint& tx, const Endpoint& rx) {
  for (auto& callback : on_connect_as_tx_callbacks_) {
    callback(tx, rx);
  }
}

void MessageLedger::connectAsRx(const Endpoint& tx, const Endpoint& rx) {
  for (auto& callback : on_connect_as_rx_callbacks_) {
    callback(tx, rx);
  }
}

}  // namespace alice
}  // namespace isaac
