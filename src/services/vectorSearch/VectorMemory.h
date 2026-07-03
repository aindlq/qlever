// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMORY_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMORY_H

#include <cstddef>
#include <cstdint>

#if defined(_SC_PHYS_PAGES) || defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#endif

// Small, dependency-free memory helpers shared by the vector-index builder and
// the reader: the physical-RAM probe used to gate RAM-residency / preloading
// (so a large flat store never drives the machine into OOM/thrash) and the
// 64-byte alignment rounding used for SIMD-friendly row strides.
namespace qlever::vector {

// Total physical RAM in bytes, or 0 if it cannot be determined (in which case
// callers must NOT assume anything fits and should skip the memory-hungry
// optimisations rather than risk OOM).
inline uint64_t totalPhysicalMemoryBytes() {
#if defined(_SC_PHYS_PAGES) && defined(_SC_PAGE_SIZE)
  long pages = sysconf(_SC_PHYS_PAGES);
  long pageSize = sysconf(_SC_PAGE_SIZE);
  if (pages > 0 && pageSize > 0) {
    return static_cast<uint64_t>(pages) * static_cast<uint64_t>(pageSize);
  }
#endif
  return 0;
}

// The cache-line / AVX-512 alignment granularity that the SIMD distance kernels
// prefer for their vector loads.
inline constexpr size_t SIMD_ALIGNMENT = 64;

// Smallest multiple of `alignment` that is >= `n` (alignment must be a power of
// two). Used to pad a row's byte length up to a SIMD-friendly stride.
inline constexpr size_t alignUp(size_t n, size_t alignment = SIMD_ALIGNMENT) {
  return (n + alignment - 1) & ~(alignment - 1);
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMORY_H
