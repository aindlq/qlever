// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H

#ifdef _WIN32

#include <io.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <utility>

#include "util/Exception.h"
#include "util/sys/windows/WindowsUndefs.h"  // for windows.h (HANDLE, ReadFile, ...)

namespace ad_utility::windows {

// Emulation of POSIX `pread` for an open file descriptor: read at an explicit
// offset without disturbing any file position. A positioned `ReadFile` on a
// synchronous handle moves that handle's position, so we read through dedicated
// handles used for positioned reads exclusively.
class PositionedReadHandle {
 private:
  // A *synchronous* handle serializes every read through it on the file
  // object's lock (`FO_SYNCHRONOUS_IO`), even positioned ones - concurrent
  // readers of one shared handle collapse onto each other (~4x slower on
  // Server 2022; hit on the hot query path where producer threads share one
  // `File`). So we spread reads over a fixed pool of synchronous handles
  // indexed by thread id. Windows hands out thread ids in multiples of four, so
  // shift off the low two bits (`tid >> 2`) or the effective pool shrinks to a
  // quarter. A fixed pool bounds open handles per file regardless of how many
  // short-lived threads touch it; a hash collision just shares a handle (the
  // rare old behavior). Synchronous handles stay on the fast-I/O cached-read
  // path (an OVERLAPPED handle would avoid the lock but force the slower IRP
  // path). Opened lazily, closed together in `close()`.
  static constexpr size_t kNumHandles = 64;
  static constexpr int kMaxRetries = 10;
  std::array<std::atomic<void*>, kNumHandles> handles_{};

 public:
  PositionedReadHandle() = default;
  ~PositionedReadHandle() { close(); }

  PositionedReadHandle(const PositionedReadHandle&) = delete;
  PositionedReadHandle& operator=(const PositionedReadHandle&) = delete;

  // Move semantics: transfer the cached handles. The moved-from object is left
  // empty but usable - its slots are null, so a later read re-opens lazily.
  PositionedReadHandle(PositionedReadHandle&& rhs) noexcept { adoptFrom(rhs); }
  PositionedReadHandle& operator=(PositionedReadHandle&& rhs) noexcept {
    if (this != &rhs) {
      close();
      adoptFrom(rhs);
    }
    return *this;
  }

  // Read up to `count` bytes at the absolute `offset` of the open file
  // descriptor `fd` (which must refer to the same open file for all calls
  // between two calls to `close()`). Returns the number of bytes read, or -1
  // on error (like POSIX `pread`).
  ssize_t pread(int fd, void* buffer, size_t count, off_t offset) {
    if (_get_osfhandle(fd) == -1) {
      errno = EBADF;
      return -1;
    }
    const HANDLE handle = getOrOpen(fd);
    // `ReadFile` takes a 32-bit length; clamp so a >=4 GiB request (which could
    // otherwise truncate to 0 and be misread as EOF) is served as a short read
    // that the caller's loop simply repeats.
    const DWORD toRead =
        static_cast<DWORD>(std::min<size_t>(count, size_t{1} << 30));
    for (int attempt = 0;; ++attempt) {
      OVERLAPPED overlapped = {};
      overlapped.Offset = static_cast<DWORD>(static_cast<uint64_t>(offset));
      overlapped.OffsetHigh =
          static_cast<DWORD>(static_cast<uint64_t>(offset) >> 32);
      DWORD numBytesRead = 0;
      if (ReadFile(handle, buffer, toRead, &numBytesRead, &overlapped)) {
        return static_cast<ssize_t>(numBytesRead);
      }
      const DWORD lastError = GetLastError();
      // A positioned `ReadFile` past EOF fails with `ERROR_HANDLE_EOF`; POSIX
      // `pread` returns 0, so mirror that (a short/zero read, not an error).
      if (lastError == ERROR_HANDLE_EOF) {
        return 0;
      }
      // Retry only genuinely transient failures (resource exhaustion under
      // heavy concurrent I/O); surface anything else immediately as -1.
      const bool transient = lastError == ERROR_NO_SYSTEM_RESOURCES ||
                             lastError == ERROR_NOT_ENOUGH_MEMORY ||
                             lastError == ERROR_WORKING_SET_QUOTA;
      if (!transient || attempt >= kMaxRetries) {
        errno = EIO;
        return -1;
      }
      Sleep(1);
    }
  }

  // Close all pooled handles. Must be called when the underlying file is
  // closed.
  void close() {
    for (auto& slot : handles_) {
      if (void* handle = slot.exchange(nullptr)) {
        CloseHandle(handle);
      }
    }
  }

 private:
  // Move the source's handle values into this object, leaving the source empty.
  void adoptFrom(PositionedReadHandle& rhs) noexcept {
    for (size_t i = 0; i < kNumHandles; ++i) {
      handles_[i].store(rhs.handles_[i].exchange(nullptr));
    }
  }

  // Return this thread's pool handle, lazily opening it on first use so that
  // concurrent readers use independent file objects instead of serializing.
  HANDLE getOrOpen(int fd) {
    std::atomic<void*>& slot =
        handles_[(GetCurrentThreadId() >> 2) % kNumHandles];
    void* handle = slot.load(std::memory_order_acquire);
    if (handle == nullptr) {
      HANDLE fresh = openHandle(fd);
      void* expected = nullptr;
      if (slot.compare_exchange_strong(expected, fresh,
                                       std::memory_order_acq_rel)) {
        handle = fresh;
      } else {
        // Another thread mapped to the same slot won the race; use its handle.
        CloseHandle(fresh);
        handle = expected;
      }
    }
    return handle;
  }

  // Open an independent read handle for the file behind `fd` via `ReOpenFile`,
  // which re-opens the same file *object* rather than the file's name. Like
  // POSIX `pread` on a `dup`-ed descriptor, this stays correct when the file
  // has been renamed since it was opened (which happens to the files of the
  // active index when a rebuilt index is swapped in at runtime, see
  // `File::duplicateForReading`).
  static HANDLE openHandle(int fd) {
    const HANDLE original = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
    AD_CONTRACT_CHECK(original != INVALID_HANDLE_VALUE && original != nullptr);
    auto reopen = [original]() {
      return ReOpenFile(original, GENERIC_READ,
                        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                        0);
    };
    HANDLE fresh = reopen();
    // Retry transient failures (e.g. under heavy system load).
    for (int i = 0; fresh == INVALID_HANDLE_VALUE && i < kMaxRetries; ++i) {
      Sleep(1);
      fresh = reopen();
    }
    AD_CONTRACT_CHECK(fresh != INVALID_HANDLE_VALUE);
    return fresh;
  }
};

// Windows positioned-read strategy for `ad_utility::File` (the override; the
// default is `ad_utility::posix::PositionedReader`). Reads are served by the
// `pread`-emulating `PositionedReadHandle` pool. See
// `util/sys/PositionedReader.h` for the compile-time seam selecting the impl.
class PositionedReader {
 private:
  // `mutable` so the const `readAtOffset` can lazily open the handle pool.
  mutable PositionedReadHandle preadHandle_;

 public:
  PositionedReader() = default;
  PositionedReader(PositionedReader&&) noexcept = default;
  PositionedReader& operator=(PositionedReader&&) noexcept = default;
  PositionedReader(const PositionedReader&) = delete;
  PositionedReader& operator=(const PositionedReader&) = delete;

  // Read through handles derived from `fd`, so reads keep working after the
  // file has been renamed. Returns bytes read, 0 at EOF, or -1 on error.
  ssize_t readAtOffset(int fd, void* buffer, size_t count, off_t offset) const {
    return preadHandle_.pread(fd, buffer, count, offset);
  }

  // Drop pooled handles; called on `close()`/re-`open()`.
  void close() { preadHandle_.close(); }
};

}  // namespace ad_utility::windows

#endif  // _WIN32
#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H
