// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
#define QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H

#include <cstdio>

#include "backports/filesystem.h"

namespace ad_utility::detail {

inline FILE* openFile(const char* filename, const char* mode) {
  return std::fopen(filename, mode);
}

template <typename... Args>
void prepareTruncatingRewrite(const ql::filesystem::path&, const Args&...) {}

}  // namespace ad_utility::detail

#endif  // QLEVER_SRC_UTIL_SYS_PORTABLEFILEOPEN_H
