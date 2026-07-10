// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_FILEMAPPING_H
#define QLEVER_SRC_UTIL_SYS_FILEMAPPING_H

namespace ad_utility {
enum class AccessPattern { None, Random, Sequential };
}  // namespace ad_utility

// Platform seam for `MmapVector`'s file mapping.
#include "util/sys/PosixFileMapping.h"

namespace ad_utility {
using FileMapping = posix::FileMapping;
}

#endif  // QLEVER_SRC_UTIL_SYS_FILEMAPPING_H
