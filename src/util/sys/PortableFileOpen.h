// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
#define QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H

#include <cstdio>
#include <filesystem>
#include <ios>
#include <type_traits>

#ifdef _WIN32
#include "util/sys/windows/FileCompat.h"
#endif

namespace ad_utility::detail {

// Open a `FILE*` with FILE_SHARE_DELETE + binary mode on Windows, plain `fopen`
// on POSIX.
inline FILE* openFilePortable(const char* filename, const char* mode) {
#ifdef _WIN32
  return windows::openWithPosixSharing(filename, mode);
#else
  return std::fopen(filename, mode);
#endif
}

// Prepare `path` for a truncating (non-append) rewrite. On Windows,
// POSIX-unlink it first: truncation fails while the file still has active
// mappings, whereas an immediate unlink detaches the name and lets the create
// recreate it (as on Linux). A no-op on POSIX. `args` are the stream's openmode
// flags, used to detect an append/at-end open (which must keep the existing
// file).
template <typename... Args>
void prepareTruncatingRewrite(const std::filesystem::path& path,
                              const Args&... args) {
#ifdef _WIN32
  auto isAppend = [](const auto& arg) {
    if constexpr (std::is_convertible_v<std::decay_t<decltype(arg)>,
                                        std::ios_base::openmode>) {
      return (static_cast<std::ios_base::openmode>(arg) &
              (std::ios::app | std::ios::ate)) != std::ios_base::openmode{};
    } else {
      return false;
    }
  };
  if (!(false || ... || isAppend(args))) {
    windows::posixDelete(path);
  }
#else
  (void)path;
  ((void)args, ...);
#endif
}

}  // namespace ad_utility::detail

#endif  // QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
