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

// Opens the native (wide on Windows) path, so non-ASCII paths work.
inline FILE* openFilePortable(const std::filesystem::path& path,
                              const char* mode) {
  return std::fopen(path.c_str(), mode);
}

// Prepare `path` for a truncating rewrite; a no-op on POSIX. `args` are the
// stream's open flags: a non-POSIX backend must inspect them and skip
// preparation for an append/at-end open.
template <typename... Args>
void prepareTruncatingRewrite(const std::filesystem::path& path,
                              const Args&... args) {
  (void)path;
  ((void)args, ...);
}

}  // namespace ad_utility::detail

#endif  // QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
