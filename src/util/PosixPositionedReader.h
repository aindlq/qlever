// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_POSIXPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_POSIXPOSITIONEDREADER_H

#ifndef _WIN32

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace ad_utility::posix {

// POSIX positioned-read strategy for `ad_utility::File` (the default; the
// Windows counterpart is `ad_utility::windows::PositionedReader`). A positioned
// read is served by `pread`, or - opt-in - from a read-only `mmap` via
// `memcpy`. The mmap fast path is EXPERIMENTAL and gated behind the
// `QLEVER_LINUX_MMAP` env var so the default Linux build keeps using `pread`;
// it lets us A/B mmap-vs-`pread` for the on-disk vocabulary. See
// `util/PositionedReader.h` for the compile-time seam selecting the impl.
class PositionedReader {
 private:
  const uint8_t* view_ = nullptr;
  size_t size_ = 0;

 public:
  PositionedReader() = default;
  ~PositionedReader() { close(); }

  PositionedReader(const PositionedReader&) = delete;
  PositionedReader& operator=(const PositionedReader&) = delete;

  PositionedReader(PositionedReader&& rhs) noexcept
      : view_{std::exchange(rhs.view_, nullptr)},
        size_{std::exchange(rhs.size_, 0)} {}
  PositionedReader& operator=(PositionedReader&& rhs) noexcept {
    if (this != &rhs) {
      close();
      view_ = std::exchange(rhs.view_, nullptr);
      size_ = std::exchange(rhs.size_, 0);
    }
    return *this;
  }

  // Read up to `count` bytes at `offset`: from the mapping via `memcpy` if one
  // is active, else via `pread` on `fd`. `filename` is unused on POSIX (a
  // mapping already holds its own reference to the file). Returns the number of
  // bytes read, 0 at/after EOF, or -1 on error (POSIX `pread` contract).
  ssize_t readAtOffset(int fd, const std::string& filename, void* buffer,
                       size_t count, off_t offset) const {
    (void)filename;
    if (view_ == nullptr) {
      return ::pread(fd, buffer, count, offset);
    }
    const size_t off = static_cast<size_t>(offset);
    if (off >= size_) return 0;
    const size_t toCopy = std::min(count, size_ - off);
    std::memcpy(buffer, view_ + off, toCopy);
    return static_cast<ssize_t>(toCopy);
  }

  // Enable the mmap fast path for `filename`. Opt-in via `QLEVER_LINUX_MMAP`;
  // a no-op otherwise, so the default build keeps using `pread`. Best-effort:
  // on any mapping failure the object stays unmapped and reads fall back to
  // `pread`.
  void enableMemoryMappedReads(const std::string& filename) {
    if (std::getenv("QLEVER_LINUX_MMAP") == nullptr) return;
    close();
    const int fd = ::open(filename.c_str(), O_RDONLY);
    if (fd < 0) return;
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
      ::close(fd);
      return;
    }
    const size_t sz = static_cast<size_t>(st.st_size);
    void* v = mmap(nullptr, sz, PROT_READ, MAP_SHARED, fd, 0);
    ::close(fd);  // the mapping keeps its own reference to the file
    if (v == MAP_FAILED) return;
    view_ = static_cast<const uint8_t*>(v);
    size_ = sz;
  }

  // Release any mapping; called on `close()`/re-`open()` of the owning `File`.
  void close() {
    if (view_ != nullptr) {
      munmap(const_cast<void*>(static_cast<const void*>(view_)), size_);
      view_ = nullptr;
    }
    size_ = 0;
  }
};

}  // namespace ad_utility::posix

#endif  // !_WIN32
#endif  // QLEVER_SRC_UTIL_POSIXPOSITIONEDREADER_H
