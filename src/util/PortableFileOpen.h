// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_PORTABLEFILEOPEN_H
#define QLEVER_SRC_UTIL_PORTABLEFILEOPEN_H

#include <cstdio>
#include <filesystem>
#include <ios>
#include <type_traits>

// Portable file-open / rewrite-prepare helpers, keeping the single `#ifdef
// _WIN32` here so `File.h` stays platform-neutral. On Windows both route
// through the FILE_SHARE_DELETE / POSIX-unlink shims in `FileCompat.h` so open
// files stay renamable/removable and index files can be rewritten while still
// mapped, exactly like a POSIX `unlink`; on POSIX they are plain libc / no-ops.
#ifdef _WIN32
#include "util/windows/FileCompat.h"
#endif

namespace ad_utility::detail {

// Open a `FILE*` with POSIX sharing semantics: FILE_SHARE_DELETE + binary mode
// on Windows, plain `fopen` on POSIX.
inline FILE* openFilePortable(const char* filename, const char* mode) {
#ifdef _WIN32
  return windows::openWithPosixSharing(filename, mode);
#else
  return std::fopen(filename, mode);
#endif
}

// Prepare `path` for a *truncating* (non-append) rewrite. On Windows, POSIX-
// unlink it first (`posixDelete`): truncation fails while the file still has
// active memory mappings, whereas an immediate unlink detaches the name and
// lets the subsequent create recreate it right away (leaving old mappings on
// the old contents, as on Linux). A no-op on POSIX. `args` are the stream's
// openmode flags, used to detect an append/at-end open (which must keep the
// existing file).
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
    windows::posixDelete(path.string().c_str());
  }
#else
  (void)path;
  ((void)args, ...);
#endif
}

}  // namespace ad_utility::detail

#endif  // QLEVER_SRC_UTIL_PORTABLEFILEOPEN_H
