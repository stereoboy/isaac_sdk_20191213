/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <map>
#include <memory>
#include <string>
#include <utility>

#include "engine/alice/component.hpp"
#include "engine/alice/components/Codelet.hpp"
#include "engine/alice/message.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac { namespace cask { class Cask; }}

namespace isaac {
namespace alice {

// Stores data for in a log file. This component can for example be used to write incoming messages
// to a log file. The messages can then be replayed using the Replay component.
//
// In order to record a message channel setup an edge from the publishing component to the Recorder
// component. The source channel is the name of the channel under which the publishing component
// publishes the data. The target channel name on the Recorder component can be choosen freely.
// When data is replayed it will be published by the Replay component under that same channel name.
//
// Warning: Please note that the log container format is not yet final and that breaking changes
// might occur in in the future.
//
// The root directory used to log data is `base_directory/exec_uuid/tag/...` where both
// `base_directory` and `tag` are configuration parameters. `exec_uuid` is a UUID which changed for
// every execution of an app and is unique over all possible executions. If `tag` is the empty
// string the root log directory is just `base_directory/exec_uuid/...`.
//
// Multiple recorders can write to the same root log directory. In this case they share the same
// key-value database. However only one recorder is allowed per log series. This means if the same
// component/key channel is logged by two different recorders they can not write to the same log
// directory.
class Recorder : public Component {
 public:
  Recorder();
  ~Recorder();
  void initialize() override;
  void start() override;
  void stop() override;
  void deinitialize() override;

  // Opens the cask container specified by configuration parameters
  void openCask();
  // Gets the cask log container
  cask::Cask* cask() const { return cask_.get(); }

  // Writes a message to the log and update the channel index
  void log(const Component* component, const std::string& key, ConstMessageBasePtr message);

  // Writes a message to the log
  void logMessage(const Component* component, ConstMessageBasePtr message);
  // Writes a new message channel entry to the log
  void logChannel(const Component* component, const std::string& key, ConstMessageBasePtr message);

  // Return the number of channels connected to this component
  size_t numChannels();

  // The base directory used as part of the log directory (see class comment)
  ISAAC_PARAM(std::string, base_directory, "/tmp/isaac")
  // A tag used as part of the log directory (see class comment)
  ISAAC_PARAM(std::string, tag, "")
  // Can be used to disable logging.
  ISAAC_PARAM(bool, enabled, true)

 private:
  // Writes an index with all channels being logged to the log container
  void writeChannelIndex();
  // Storage for messages on a channel
  std::unique_ptr<cask::Cask> cask_;
  // A mapping from component channels to series uuids used to store the messages
  std::map<std::pair<Uuid, std::string>, Uuid> component_key_to_uuid_;
};

}  // namespace alice
}  // namespace isaac

ISAAC_ALICE_REGISTER_COMPONENT(isaac::alice::Recorder)
