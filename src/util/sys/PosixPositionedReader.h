// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

namespace ad_utility::posix {

// POSIX positioned reads for `File`: `pread` by default, or - opt-in via the
// `QLEVER_LINUX_MMAP` environment variable - a read-only `mmap` served by
// `memcpy`. The mapped path is experimental; it lets us A/B-test mmap against
// `pread` for the on-disk vocabulary's many small random reads.
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

  // Read up to `count` bytes at `offset` from the mapping via `memcpy` if one
  // is active, else via `pread` on `fd`. `filename` is unused on POSIX. Returns
  // the number of bytes read, 0 at EOF, -1 on error.
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

  // Enable the mmap fast path for `filename`, opt-in via `QLEVER_LINUX_MMAP`
  // (else a no-op). Best-effort: on any failure reads fall back to `pread`.
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
    ::close(fd);
    if (v == MAP_FAILED) return;
    view_ = static_cast<const uint8_t*>(v);
    size_ = sz;
  }

  // Release any mapping; called on `close`/re-`open` of the owning `File`.
  void close() {
    if (view_ != nullptr) {
      munmap(const_cast<void*>(static_cast<const void*>(view_)), size_);
      view_ = nullptr;
    }
    size_ = 0;
  }
};

}  // namespace ad_utility::posix

#endif  // QLEVER_SRC_UTIL_SYS_POSIXPOSITIONEDREADER_H
