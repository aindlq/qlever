//  Copyright 2026, University of Freiburg,
//  Chair of Algorithms and Data Structures.

#include "util/JemallocLibraryHooks.h"

#if defined(_WIN32) && defined(QLEVER_JEMALLOC_LIBRARY_HOOKS)

#include <unicode/uclean.h>
#include <unicode/utypes.h>

#include <cstddef>

// jemalloc's C API is `je_`-prefixed on Windows/MinGW (see the jemalloc block
// in the top-level CMakeLists). We declare the three entry points we use here
// instead of pulling in <jemalloc/jemalloc.h>, so this translation unit does
// not depend on jemalloc's include path; the linker resolves them against the
// front-linked jemalloc archive.
extern "C" {
void* je_malloc(std::size_t size);
void* je_realloc(void* ptr, std::size_t size);
void je_free(void* ptr);
}

namespace {
// ICU allocator callbacks that forward to jemalloc's `je_`-prefixed C API.
void* U_CALLCONV icuJemallocMalloc(const void*, size_t size) {
  return je_malloc(size);
}
void* U_CALLCONV icuJemallocRealloc(const void*, void* mem, size_t size) {
  return je_realloc(mem, size);
}
void U_CALLCONV icuJemallocFree(const void*, void* mem) { je_free(mem); }
}  // namespace

namespace ad_utility {
void installLibraryAllocatorHooks() {
  UErrorCode status = U_ZERO_ERROR;
  u_setMemoryFunctions(nullptr, &icuJemallocMalloc, &icuJemallocRealloc,
                       &icuJemallocFree, &status);
  // A failure here only means ICU keeps its default (UCRT) allocator, which is
  // harmless for correctness, so `status` is intentionally not checked.
}
}  // namespace ad_utility

#else

namespace ad_utility {
// No-op: on non-Windows platforms `malloc` already routes to jemalloc, and when
// the hooks are disabled we deliberately keep ICU on its default allocator.
void installLibraryAllocatorHooks() {}
}  // namespace ad_utility

#endif
