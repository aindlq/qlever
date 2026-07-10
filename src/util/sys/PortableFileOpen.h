// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
#define QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H

#include <cstdio>
#include <filesystem>

namespace ad_utility::detail {

inline FILE* openFilePortable(const char* filename, const char* mode) {
  return std::fopen(filename, mode);
}

// Prepare `path` for a truncating (non-append) rewrite; a no-op on POSIX.
template <typename... Args>
void prepareTruncatingRewrite(const std::filesystem::path& path,
                              const Args&... args) {
  (void)path;
  ((void)args, ...);
}

}  // namespace ad_utility::detail

#endif  // QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
