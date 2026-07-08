//  Copyright 2026, University of Freiburg,
//  Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_JEMALLOCLIBRARYHOOKS_H
#define QLEVER_SRC_UTIL_JEMALLOCLIBRARYHOOKS_H

namespace ad_utility {
// Route the allocations of bundled C libraries (currently ICU) to jemalloc.
//
// On Windows/MinGW jemalloc's C API is `je_`-prefixed, so plain `malloc`/`free`
// - and hence a C library such as ICU that allocates through them - stay on the
// UCRT heap instead of jemalloc's arenas. This installs jemalloc-backed
// allocation functions for ICU (via `u_setMemoryFunctions`), so vocabulary
// collation allocates from the same arena as the rest of QLever. It must be
// called before any other ICU use, i.e. as the first thing in `main`.
//
// On other platforms `malloc` already IS jemalloc (via ELF symbol
// interposition), so no per-library hook is needed and this is a no-op. The
// effect is additionally gated behind the `QLEVER_JEMALLOC_LIBRARY_HOOKS` CMake
// option so it can be toggled for benchmarking.
void installLibraryAllocatorHooks();
}  // namespace ad_utility

#endif  // QLEVER_SRC_UTIL_JEMALLOCLIBRARYHOOKS_H
