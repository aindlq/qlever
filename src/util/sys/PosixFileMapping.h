// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem Kozlov <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_UTIL_SYS_POSIXFILEMAPPING_H
#define QLEVER_SRC_UTIL_SYS_POSIXFILEMAPPING_H

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <string>

#include "util/Exception.h"
#include "util/sys/AccessPattern.h"

namespace ad_utility::posix {

// The POSIX file-mapping primitive used by `MmapVector`. Stateless: the mapped
// pointer and its size are owned by the `MmapVector` and passed back in.
class FileMapping {
 public:
  // Persist the mapped data and release the mapping. POSIX can resize the
  // backing file while it is still mapped, so the metadata is written first.
  template <typename Persist, typename Unmap>
  void persistAndUnmap(const Persist& persist, const Unmap& unmap) const {
    persist();
    unmap();
  }

  // Map `bytesize` bytes of `filename` (`MAP_SHARED`) and return the address.
  void* map(const std::string& filename, size_t bytesize, bool writable) {
    // open to get valid file descriptor
    const int fd = ::open(filename.c_str(), writable ? O_RDWR : O_RDONLY);
    const int prot = writable ? (PROT_READ | PROT_WRITE) : PROT_READ;
    // map_shared because we need our updates in the original file
    void* ptr = mmap(nullptr, bytesize, prot, MAP_SHARED, fd, 0);
    AD_CONTRACT_CHECK(ptr != MAP_FAILED);
    // the filedescriptor and thus our mapping will still be valid
    // after closing, because mmap increases the count by one
    ::close(fd);
    return ptr;
  }

  void unmap(void* ptr, size_t bytesize) { munmap(ptr, bytesize); }

  // Grow/shrink a mapping in place (`mremap`); Linux-only, callers guard on it.
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

  // The constants `MADV_SEQUENTIAL` etc. don't seem to be present on all POSIX
  // systems, in particular they are not present on the `QNX` platform which
  // we target with the `REDUCED_FEATURE_SET` mode. Therefore, we simply disable
  // the following `madvise` calls, as they only are hints to the runtime and
  // do not change the program semantics.
  void advise(void* ptr, size_t bytesize, AccessPattern pattern) {
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

#endif  // QLEVER_SRC_UTIL_SYS_POSIXFILEMAPPING_H
