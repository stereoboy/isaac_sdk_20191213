/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "engine/core/array/byte_array.hpp"
#include "engine/core/byte.hpp"
#include "engine/gems/uuid/uuid.hpp"

namespace isaac {
namespace cask {

// A storage container for arbitrary data blobs. It supports two different kinds of storage:
// key-value storage using an LMDB database as a backend, and series storage which stores data
// sequentially in a file stream. Key-value storage is good for unstructured data, while series
// storage is good for storing data from a continuous source stream.
class Cask {
 public:
  // Access mode in which to open the storage.
  enum class Mode {
    Read,
    Write
  };

  // Creates and opens a casks
  Cask(const std::string& directory, Mode mode);
  // Default constructor
  Cask();
  // Closes and destroys a cask
  ~Cask();

  // Opens an existing cask or creates a new cask at the given directory
  void open(const std::string& directory, Mode mode);
  // Closes the cask if it was opened
  void close();
  // Get this cask root directory
  std::string getRoot() const { return root_; }

  // Writes a blob to key-value storage
  void keyValueWrite(const Uuid& key, const ByteArrayConstView& blob);
  void keyValueWrite(const Uuid& key, const std::vector<byte>& blob);
  // Writes a list of blobs sequentially to key-value storage
  void keyValueWrite(const Uuid& key, const std::vector<ByteArrayConstView>& blobs);
  // Writes a data block with given length to key-value storage. The given callback will be called
  // with the target storage space and the user is expected to copy the data he wants to store.
  void keyValueWrite(const Uuid& key, size_t length, std::function<void(byte*, byte*)> on_write);
  // Reads a blob from key-value storage into the given buffer. Resizes the buffer if necessary.
  void keyValueRead(const Uuid& key, std::vector<byte>& blob);
  // Reads a blob from key-value storage. The given callback will be called with the source
  // storage space, and the user is expected to read the data. Storage space will not stay
  // accessible after the callback returns.
  void keyValueRead(const Uuid& key, std::function<void(const byte*, const byte*)> on_read);

  // Opens a series with the given value size. The `value_size` is the size of values which will
  // be written to the series.
  void seriesOpen(const Uuid& uuid, int value_size = -1);
  // Closes a series
  void seriesClose(const Uuid& uuid);
  // Appends a blob to a series
  void seriesAppend(const Uuid& uuid, const ByteArrayConstView& blob);
  void seriesAppend(const Uuid& uuid, const byte* begin, const byte* end);
  // Reads the number of values and the size of each value for a series
  void seriesDimensions(const Uuid& uuid, size_t& count, size_t& value_size);
  // Reads the `index`-th value from a series into the given buffer. Resizes buffer if necessary.
  void seriesRead(const Uuid& uuid, size_t index, std::vector<byte>& blob);

 private:
  class KeyValueImpl;
  class SeriesImpl;

  SeriesImpl* getSeries(const Uuid& uuid);
  SeriesImpl* createSeries(const Uuid& uuid);
  std::unique_ptr<Cask::SeriesImpl> removeSeries(const Uuid& uuid);

  // Specifies if this log is opened for write access. Read and write are exclusive.
  bool is_writing_;
  // root directory of the cask
  std::string root_;

  std::unique_ptr<KeyValueImpl> kv_impl_;
  std::map<Uuid, std::unique_ptr<SeriesImpl>> series_impl_;
  std::mutex series_impl_mutex_;
  std::mutex series_file_mutex_;
};

}  // namespace cask
}  // namespace isaac
