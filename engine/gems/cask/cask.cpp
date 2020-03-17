/*
Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.

NVIDIA CORPORATION and its licensors retain all intellectual property
and proprietary rights in and to this software, related documentation
and any modifications thereto. Any use, reproduction, disclosure or
distribution of this software and related documentation without an express
license agreement from NVIDIA CORPORATION is strictly prohibited.
*/
#include "cask.hpp"

#include <algorithm>
#include <experimental/filesystem>  // or #include <filesystem>  // NOLINT
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "engine/core/assert.hpp"
#include "engine/gems/serialization/blob.hpp"
#include "liblmdb/lmdb.h"

// Size of the memory allocated for lmdb.
constexpr size_t kLmdbMapSize = 100ull*1024ull*1024ull*1024ull;

namespace isaac {
namespace cask {

namespace {

constexpr int kMaxReaders = 16;

// Creates a directory for a file in case it does not exist already
void CreateDirectoriesForFile(const std::string& filename) {
  const auto dir = std::experimental::filesystem::path(filename).parent_path();
  std::experimental::filesystem::create_directories(dir);
}

}  // namespace

class Cask::KeyValueImpl {
 public:
  void checkError(int rc) {
    if (rc != MDB_SUCCESS) {
      if (lmdb_env_ != nullptr) {
        mdb_env_close(lmdb_env_);
      }
    }
    ASSERT(rc != MDB_NOTFOUND, "Trying to load a non-existing channel from cask");
    ASSERT(rc == MDB_SUCCESS, "Error %d: %s", rc, mdb_strerror(rc));
  }

  void open(const std::string& filename, bool is_writing) {
    if (is_writing) {
      LOG_INFO("Creating new log '%s' for writing", filename.c_str());
    } else {
      LOG_INFO("Opening log '%s' for reading", filename.c_str());
    }
    std::unique_lock<std::mutex> lock(mutex);
    CreateDirectoriesForFile(filename);
    checkError(mdb_env_create(&lmdb_env_));
    checkError(mdb_env_set_maxreaders(lmdb_env_, kMaxReaders));
    checkError(mdb_env_set_mapsize(lmdb_env_, kLmdbMapSize));
    checkError(mdb_env_open(lmdb_env_, filename.c_str(),
                            MDB_NOSUBDIR | (is_writing ? 0 : MDB_RDONLY), 0664));
  }

  void close() {
    std::unique_lock<std::mutex> lock(mutex);
    if (lmdb_txn_ != nullptr) {
      mdb_txn_abort(lmdb_txn_);
      lmdb_txn_ = nullptr;
    }
    if (lmdb_env_ != nullptr) {
      mdb_dbi_close(lmdb_env_, lmdb_dbi_);
      mdb_env_close(lmdb_env_);
      lmdb_env_ = nullptr;
    }
  }

  void write(const Uuid& key, size_t length, bool allow_overwrite,
             std::function<void(byte*, byte*)> on_write) {
    std::unique_lock<std::mutex> lock(mutex);
    // start a new transaction
    checkError(mdb_txn_begin(lmdb_env_, NULL, 0, &lmdb_txn_));
    checkError(mdb_dbi_open(lmdb_txn_, NULL, 0, &lmdb_dbi_));
    // prepare key and value for the put operation
    std::array<uint64_t, 2> keybytes{key.lower(), key.upper()};
    MDB_val mkey;
    mkey.mv_size = 16;
    mkey.mv_data = reinterpret_cast<void*>(&keybytes[0]);
    MDB_val mvalue;
    mvalue.mv_size = length;
    mvalue.mv_data = nullptr;
    // reserve a space where we can write data in the database
    const int rc = mdb_put(lmdb_txn_, lmdb_dbi_, &mkey, &mvalue,
                           (allow_overwrite ? 0 : MDB_NOOVERWRITE) | MDB_RESERVE);
    // Don't treat duplicates as an error
    if (!allow_overwrite && rc == MDB_KEYEXIST) {
      mdb_txn_abort(lmdb_txn_);
      lmdb_txn_ = nullptr;
      return;
    }
    checkError(rc);
    // copy the data over
    ASSERT(mvalue.mv_data, "target pointer must not be null");
    byte* dst = reinterpret_cast<byte*>(mvalue.mv_data);
    on_write(dst, dst + length);
    // finish the transaction
    checkError(mdb_txn_commit(lmdb_txn_));
    lmdb_txn_ = nullptr;
  }

  void read(const Uuid& key, std::function<void(const byte*, const byte*)> on_read) {
    std::unique_lock<std::mutex> lock(mutex);
    // start a new transaction
    checkError(mdb_txn_begin(lmdb_env_, NULL, MDB_RDONLY, &lmdb_txn_));
    checkError(mdb_dbi_open(lmdb_txn_, NULL, 0, &lmdb_dbi_));
    // prepare key and value for the put operation
    std::array<uint64_t, 2> keybytes{key.lower(), key.upper()};
    MDB_val mkey;
    mkey.mv_size = 16;
    mkey.mv_data = reinterpret_cast<void*>(&keybytes[0]);
    MDB_val mvalue;
    mvalue.mv_size = 0;
    mvalue.mv_data = nullptr;
    const int rc = mdb_get(lmdb_txn_, lmdb_dbi_, &mkey, &mvalue);
    checkError(rc);
    // copy the data over
    ASSERT(mvalue.mv_data, "target pointer must not be null");
    const byte* src = reinterpret_cast<const byte*>(mvalue.mv_data);
    on_read(src, src + mvalue.mv_size);
    // finish the transaction
    checkError(mdb_txn_commit(lmdb_txn_));
    lmdb_txn_ = nullptr;
  }

 private:
  // LMDB key-value database
  std::mutex mutex;
  MDB_env* lmdb_env_ = nullptr;
  MDB_txn* lmdb_txn_ = nullptr;
  unsigned int lmdb_dbi_;  // type is MDB_dbi, but we don't want to include lmdb.h in the header
};

class Cask::SeriesImpl {
 public:
  void open(const std::string& filename, size_t value_size, bool is_writing) {
    std::unique_lock<std::mutex> lock(mutex_);
    ASSERT(!file_.is_open(), "Already openend file '%s'", filename.c_str());
    if (is_writing) {
      CreateDirectoriesForFile(filename);
    }
    file_.open(filename, (is_writing ? std::ios_base::out : std::ios_base::in)
                         | std::ios_base::binary);
    ASSERT(file_.is_open(), "Could not open file '%s'", filename.c_str());
    // Write/read the element size
    if (is_writing) {
      ASSERT(value_size > 0, "Series element size must be greater than 0");
      value_size_ = value_size;
      file_.write(reinterpret_cast<const char*>(&value_size_), 4);
      file_.flush();  // FIXME Flusing each time might be slow but otherwise we loose data on crash.
      count_ = 0;
    } else {
      file_.read(reinterpret_cast<char*>(&value_size_), 4);
      // TODO should not ignore the given value_size
      // get the number of elements
      const std::streampos fpos_start = file_.tellg();
      file_.seekg(0, std::ios::end);
      const int64_t fsize = file_.tellg() - fpos_start;
      ASSERT(fsize % value_size_ == 0, "Wrong file size: %zu %zu", fsize, value_size_);
      count_ = fsize / value_size_;
    }
  }

  void close() {
    std::unique_lock<std::mutex> lock(mutex_);
    file_.close();
  }

  void append(const byte* begin, size_t length) {
    std::unique_lock<std::mutex> lock(mutex_);
    ASSERT(length == value_size_, "Invalid blob size: %zu vs %zu", length, value_size_);
    file_.write(reinterpret_cast<const char*>(begin), length);
    file_.flush();  // FIXME Flusing each time might be slow but otherwise we loose data on crash.
    checkFile();
    count_++;
  }

  void read(size_t index, std::vector<byte>& blob) {
    ASSERT(index < count_, "Index is out of range");
    std::unique_lock<std::mutex> lock(mutex_);
    file_.seekg(4 + value_size_ * index);
    checkFile();
    blob.resize(value_size_);
    file_.read(reinterpret_cast<char*>(blob.data()), value_size_);
    checkFile();
  }

  void dimensions(size_t& count, size_t& value_size) {
    std::unique_lock<std::mutex> lock(mutex_);
    value_size = value_size_;
    count = count_;
  }

 private:
  void checkFile() {
    ASSERT(file_.good(), "Could not write bytes to series. iostate=%d (bad=%d, eof=%d, fail=%d)",
           file_.rdstate(),
           (file_.rdstate() & std::ios_base::badbit) > 0 ? 1 : 0,
           (file_.rdstate() & std::ios_base::eofbit) > 0 ? 1 : 0,
           (file_.rdstate() & std::ios_base::failbit) > 0 ? 1 : 0);
  }

  size_t value_size_;
  size_t count_;
  std::mutex mutex_;
  std::fstream file_;
};

Cask::Cask() { }

Cask::Cask(const std::string& directory, Cask::Mode mode) {
  open(directory, mode);
}

Cask::~Cask() {
  close();
  kv_impl_.reset();
}

void Cask::open(const std::string& directory, Cask::Mode mode) {
  ASSERT(root_ == "", "Log already open");
  root_ = directory;
  is_writing_ = mode == Mode::Write;
  kv_impl_.reset(new KeyValueImpl());
  kv_impl_->open(root_ + "/kv", is_writing_);
}

void Cask::close() {
  if (kv_impl_) {
    kv_impl_->close();
    kv_impl_.reset();
  }
  root_ = "";
}

void Cask::keyValueWrite(const Uuid& key, const ByteArrayConstView& blob) {
  ASSERT(is_writing_, "Cask is read only");
  kv_impl_->write(key, blob.size(), true,
      [&] (byte* begin, byte* end) {
        ASSERT(begin + blob.size() == end, "out of bounds");
        std::copy(blob.begin(), blob.end(), begin);
      });
}

void Cask::keyValueWrite(const Uuid& key, const std::vector<byte>& blob) {
  ASSERT(is_writing_, "Cask is read only");
  kv_impl_->write(key, blob.size(), true,
      [&] (byte* begin, byte* end) {
        ASSERT(begin + blob.size() == end, "out of bounds");
        std::copy(blob.begin(), blob.end(), begin);
      });
}

void Cask::keyValueWrite(const Uuid& key, const std::vector<ByteArrayConstView>& blobs) {
  ASSERT(is_writing_, "Cask is read only");
  const size_t length = serialization::AccumulateLength(blobs);
  kv_impl_->write(key, length, true,
      [&] (byte* begin, byte* end) {
        ASSERT(begin + length == end, "out of bounds");
        serialization::CopyAll(blobs, begin, end);
      });
}

void Cask::keyValueWrite(const Uuid& key, size_t length,
                         std::function<void(uint8_t*, uint8_t*)> on_write) {
  ASSERT(is_writing_, "Cask is read only");
  kv_impl_->write(key, length, true, std::move(on_write));
}

void Cask::keyValueRead(const Uuid& key, std::vector<byte>& blob) {
  ASSERT(!is_writing_, "Cask is write only");
  kv_impl_->read(key,
      [&](const byte* begin, const byte* end) {
        blob.resize(std::distance(begin, end));
        std::copy(begin, end, blob.begin());
      });
}

void Cask::keyValueRead(const Uuid& key, std::function<void(const byte*, const byte*)> on_read) {
  ASSERT(!is_writing_, "Cask is write only");
  kv_impl_->read(key, std::move(on_read));
}

void Cask::seriesOpen(const Uuid& uuid, int value_size) {
  // Prevent multiple open/close happening at the same time to avoid that we open a series while
  // we are closing it, or other strange things.
  std::unique_lock<std::mutex> lock(series_file_mutex_);
  createSeries(uuid)->open(root_ + "/" + uuid.str(), value_size, is_writing_);
}

void Cask::seriesClose(const Uuid& uuid) {
  // Prevent multiple open/close happening at the same time to avoid that we open a series while
  // we are closing it, or other strange things.
  std::unique_lock<std::mutex> lock(series_file_mutex_);
  removeSeries(uuid)->close();
}

void Cask::seriesAppend(const Uuid& uuid, const ByteArrayConstView& blob) {
  ASSERT(is_writing_, "Cask is read only");
  getSeries(uuid)->append(blob.begin(), blob.size());
}

void Cask::seriesAppend(const Uuid& uuid, const byte* begin, const byte* end) {
  ASSERT(is_writing_, "Cask is read only");
  getSeries(uuid)->append(begin, std::distance(begin, end));
}

void Cask::seriesRead(const Uuid& uuid, size_t index, std::vector<byte>& blob) {
  ASSERT(!is_writing_, "Cask is write only");
  getSeries(uuid)->read(index, blob);
}

void Cask::seriesDimensions(const Uuid& uuid, size_t& count, size_t& value_size) {
  getSeries(uuid)->dimensions(count, value_size);
}

Cask::SeriesImpl* Cask::getSeries(const Uuid& uuid) {
  std::unique_lock<std::mutex> lock(series_impl_mutex_);
  auto it = series_impl_.find(uuid);
  ASSERT(it != series_impl_.end(), "Series does not exist");
  return it->second.get();
}

Cask::SeriesImpl* Cask::createSeries(const Uuid& uuid) {
  std::unique_lock<std::mutex> lock(series_impl_mutex_);
  auto it = series_impl_.find(uuid);
  ASSERT(it == series_impl_.end(), "Series already exists");
  auto series_uptr = std::make_unique<SeriesImpl>();
  auto* series = series_uptr.get();
  series_impl_.emplace(uuid, std::move(series_uptr));
  return series;
}

std::unique_ptr<Cask::SeriesImpl> Cask::removeSeries(const Uuid& uuid) {
  std::unique_lock<std::mutex> lock(series_impl_mutex_);
  auto it = series_impl_.find(uuid);
  ASSERT(it != series_impl_.end(), "Series does not exist");
  std::unique_ptr<Cask::SeriesImpl> uptr(it->second.release());
  series_impl_.erase(it);
  return uptr;
}

}  // namespace cask
}  // namespace isaac
