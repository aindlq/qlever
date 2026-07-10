// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem Kozlov <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

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
