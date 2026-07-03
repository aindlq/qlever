// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_POSIXFILEMAPPING_H
#define QLEVER_SRC_UTIL_POSIXFILEMAPPING_H

#ifndef _WIN32

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdint>
#include <string>

#include "util/Exception.h"

namespace ad_utility::posix {

// The POSIX file-mapping primitive used by `MmapVector` (the default; the
// Windows counterpart is `ad_utility::windows::FileMapping`). It is stateless:
// the mapped pointer and its size are owned by the `MmapVector`, which passes
// them back into the `unmap`/`advise`/`remap` calls. See `util/FileMapping.h`
// for the compile-time seam that selects between the two implementations.
class FileMapping {
 public:
  // POSIX can `truncate` a file that still has active mappings, so the
  // `MmapVector` may resize before unmapping (unlike Windows).
  static constexpr bool kMustUnmapBeforeResize = false;

  // Map `bytesize` bytes of `filename` (read-only, or read/write with
  // `MAP_SHARED` so updates hit the file) and return the mapped address.
  void* map(const std::string& filename, size_t bytesize, bool writable) {
    const int fd = ::open(filename.c_str(), writable ? O_RDWR : O_RDONLY);
    const int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
    void* ptr = mmap(nullptr, bytesize, prot, MAP_SHARED, fd, 0);
    AD_CONTRACT_CHECK(ptr != MAP_FAILED);
    // The mapping keeps its own reference to the file, so the fd can be closed.
    ::close(fd);
    return ptr;
  }

  // Release a mapping previously returned by `map`.
  void unmap(void* ptr, size_t bytesize) { munmap(ptr, bytesize); }

  // Grow/shrink an existing mapping in place (Linux `mremap`). Only ever called
  // on Linux (`MmapVector::adaptCapacity` guards the call with `#ifdef
  // __linux__`); other POSIX systems unmap and remap instead.
  void* remap(void* ptr, size_t oldBytesize, size_t newBytesize) {
#ifdef __linux__
    void* newPtr = mremap(ptr, oldBytesize, newBytesize, MREMAP_MAYMOVE);
    AD_CONTRACT_CHECK(newPtr != MAP_FAILED);
    return newPtr;
#else
    (void)ptr;
    (void)oldBytesize;
    (void)newBytesize;
    AD_FAIL();
#endif
  }

  // Advise the kernel about the access pattern (a hint only).
  void advise(void* ptr, size_t bytesize, AccessPattern pattern) {
    // `MADV_SEQUENTIAL` etc. are not present on all POSIX systems (in
    // particular not on QNX, which we target via `REDUCED_FEATURE_SET`); the
    // calls are hints only, so disable them there.
#ifndef QLEVER_REDUCED_FEATURE_SET_FOR_CPP17
    switch (pattern) {
      case AccessPattern::Sequential:
        madvise(ptr, bytesize, MADV_SEQUENTIAL);
        break;
      case AccessPattern::Random:
        madvise(ptr, bytesize, MADV_RANDOM);
        break;
      default:
        madvise(ptr, bytesize, MADV_NORMAL);
        break;
    }
#else
    (void)ptr;
    (void)bytesize;
    (void)pattern;
#endif
  }

  // Mappings must be aligned to (a multiple of) this granularity.
  static size_t pageSize() { return getpagesize(); }

  // Resize the backing file by path (POSIX `truncate`). Returns 0 on success.
  static int resizeFile(const char* path, int64_t length) {
    return ::truncate(path, length);
  }
};

}  // namespace ad_utility::posix

#endif  // !_WIN32
#endif  // QLEVER_SRC_UTIL_POSIXFILEMAPPING_H
