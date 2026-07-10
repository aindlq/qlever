// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H

#ifdef _WIN32

#include <sys/types.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <utility>

#include "util/Exception.h"
#include "util/sys/windows/WindowsUndefs.h"  // for windows.h (HANDLE, ReadFile, ...)

namespace ad_utility::windows {

// Emulation of POSIX `pread` for a named file: read at an explicit offset
// without disturbing any file position. A positioned `ReadFile` on a
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

  // Read up to `count` bytes at the absolute `offset` of the file named
  // `filename` (which must denote the same file for all calls between two
  // calls to `close()`). Returns the number of bytes read, or -1 on error
  // (like POSIX `pread`).
  ssize_t pread(const std::filesystem::path& filename, void* buffer,
                size_t count, off_t offset) {
    const HANDLE handle = getOrOpen(filename);
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
  HANDLE getOrOpen(const std::filesystem::path& filename) {
    std::atomic<void*>& slot =
        handles_[(GetCurrentThreadId() >> 2) % kNumHandles];
    void* handle = slot.load(std::memory_order_acquire);
    if (handle == nullptr) {
      HANDLE fresh = openHandle(filename);
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

  static HANDLE openHandle(const std::filesystem::path& filename) {
    HANDLE fresh =
        CreateFileW(filename.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    // Retry transient failures (e.g. under heavy system load).
    for (int i = 0; fresh == INVALID_HANDLE_VALUE && i < kMaxRetries; ++i) {
      Sleep(1);
      fresh =
          CreateFileW(filename.c_str(), GENERIC_READ,
                      FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                      nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    }
    AD_CONTRACT_CHECK(fresh != INVALID_HANDLE_VALUE);
    return fresh;
  }
};

// Memory-map a file read-only and serve positioned reads (POSIX `pread`
// semantics) from the mapped view via `memcpy`. Every warm cached `ReadFile`
// still pays a kernel round-trip (~850 ns per call on the vocabulary's
// ascending scan, up to ~4 us random), whereas a `memcpy` from a mapped view is
// tens of nanoseconds - a big win for string-heavy queries. Intended for files
// that fit in RAM (the vocabulary case).
class MemoryMappedReadFile {
 private:
  HANDLE fileHandle_ = nullptr;
  HANDLE mappingHandle_ = nullptr;
  const uint8_t* view_ = nullptr;
  uint64_t size_ = 0;

 public:
  MemoryMappedReadFile() = default;
  ~MemoryMappedReadFile() { close(); }

  MemoryMappedReadFile(const MemoryMappedReadFile&) = delete;
  MemoryMappedReadFile& operator=(const MemoryMappedReadFile&) = delete;

  MemoryMappedReadFile(MemoryMappedReadFile&& rhs) noexcept
      : fileHandle_{std::exchange(rhs.fileHandle_, nullptr)},
        mappingHandle_{std::exchange(rhs.mappingHandle_, nullptr)},
        view_{std::exchange(rhs.view_, nullptr)},
        size_{std::exchange(rhs.size_, 0)} {}
  MemoryMappedReadFile& operator=(MemoryMappedReadFile&& rhs) noexcept {
    close();
    fileHandle_ = std::exchange(rhs.fileHandle_, nullptr);
    mappingHandle_ = std::exchange(rhs.mappingHandle_, nullptr);
    view_ = std::exchange(rhs.view_, nullptr);
    size_ = std::exchange(rhs.size_, 0);
    return *this;
  }

  // Map `filename` read-only. Returns true on success; on any failure (empty
  // file, mapping unsupported) returns false and leaves the object unmapped so
  // the caller falls back to the `ReadFile` path. Full sharing (incl.
  // FILE_SHARE_DELETE) so the file stays renamable/removable while mapped.
  bool map(const std::filesystem::path& filename) {
    close();
    HANDLE fh =
        CreateFileW(filename.c_str(), GENERIC_READ,
                    FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                    nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (fh == INVALID_HANDLE_VALUE) return false;
    LARGE_INTEGER li;
    if (!GetFileSizeEx(fh, &li) || li.QuadPart == 0) {
      // `CreateFileMapping` rejects zero-length files; fall back to `pread`.
      CloseHandle(fh);
      return false;
    }
    HANDLE mh = CreateFileMappingA(fh, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mh == nullptr) {
      CloseHandle(fh);
      return false;
    }
    const void* v = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (v == nullptr) {
      CloseHandle(mh);
      CloseHandle(fh);
      return false;
    }
    fileHandle_ = fh;
    mappingHandle_ = mh;
    view_ = static_cast<const uint8_t*>(v);
    size_ = static_cast<uint64_t>(li.QuadPart);
    // Hint the OS to fault the whole mapping in as a few coalesced sequential
    // reads (the Windows analog of `madvise(MADV_WILLNEED)`), mainly helping
    // the cold case. Advisory/best-effort. `PrefetchVirtualMemory` is Windows 8
    // / Server 2012+; resolve it dynamically (and declare the range struct
    // locally) so the build doesn't depend on `_WIN32_WINNT`, skipping it on
    // older systems.
    struct MemRangeEntry {
      void* virtualAddress;
      SIZE_T numberOfBytes;
    };
    using PrefetchFn = BOOL(WINAPI*)(HANDLE, ULONG_PTR, MemRangeEntry*, ULONG);
    static const auto prefetch = reinterpret_cast<PrefetchFn>(::GetProcAddress(
        ::GetModuleHandleA("kernel32.dll"), "PrefetchVirtualMemory"));
    if (prefetch != nullptr) {
      MemRangeEntry range{const_cast<void*>(v), static_cast<SIZE_T>(size_)};
      prefetch(::GetCurrentProcess(), 1, &range, 0);
    }
    return true;
  }

  bool isMapped() const { return view_ != nullptr; }

  // POSIX-`pread`-like read from the mapped view: copy up to `count` bytes at
  // `offset`, clamped to the file size (a read at/after EOF returns 0).
  ssize_t pread(void* buffer, size_t count, off_t offset) const {
    const uint64_t off = static_cast<uint64_t>(offset);
    if (off >= size_) return 0;
    const size_t toCopy =
        static_cast<size_t>(std::min<uint64_t>(count, size_ - off));
    std::memcpy(buffer, view_ + off, toCopy);
    return static_cast<ssize_t>(toCopy);
  }

  void close() {
    if (view_ != nullptr) {
      UnmapViewOfFile(const_cast<void*>(static_cast<const void*>(view_)));
      view_ = nullptr;
    }
    if (mappingHandle_ != nullptr) {
      CloseHandle(mappingHandle_);
      mappingHandle_ = nullptr;
    }
    if (fileHandle_ != nullptr) {
      CloseHandle(fileHandle_);
      fileHandle_ = nullptr;
    }
    size_ = 0;
  }
};

// Windows positioned-read strategy for `ad_utility::File` (the override; the
// default is `ad_utility::posix::PositionedReader`). Positioned reads are
// served from a memory-mapped view via `memcpy` when one is enabled (far faster
// for the on-disk vocabulary's many small random reads), otherwise by the
// `pread`- emulating `PositionedReadHandle` pool. See
// `util/sys/PositionedReader.h` for the compile-time seam selecting the impl.
class PositionedReader {
 private:
  // `mutable` so the const `readAtOffset` can lazily open the handle pool.
  mutable PositionedReadHandle preadHandle_;
  MemoryMappedReadFile mmapRead_;

 public:
  PositionedReader() = default;
  PositionedReader(PositionedReader&&) noexcept = default;
  PositionedReader& operator=(PositionedReader&&) noexcept = default;
  PositionedReader(const PositionedReader&) = delete;
  PositionedReader& operator=(const PositionedReader&) = delete;

  // Read up to `count` bytes at `offset` of the file named `filename`: from the
  // mapping via `memcpy` if enabled, else via the positioned-`ReadFile` handle
  // pool. `fd` is unused on Windows. Returns bytes read, 0 at EOF, -1 on error.
  ssize_t readAtOffset(int fd, const std::filesystem::path& filename,
                       void* buffer, size_t count, off_t offset) const {
    (void)fd;
    return mmapRead_.isMapped()
               ? mmapRead_.pread(buffer, count, offset)
               : preadHandle_.pread(filename, buffer, count, offset);
  }

  // Enable the always-on (once mapped) mmap fast path for `filename`.
  void enableMemoryMappedReads(const std::filesystem::path& filename) {
    mmapRead_.map(filename);
  }

  // Drop the mapping and pooled handles; called on `close()`/re-`open()`.
  void close() {
    preadHandle_.close();
    mmapRead_.close();
  }
};

}  // namespace ad_utility::windows

#endif  // _WIN32
#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSPOSITIONEDREADER_H
