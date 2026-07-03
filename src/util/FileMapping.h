// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_FILEMAPPING_H
#define QLEVER_SRC_UTIL_FILEMAPPING_H

namespace ad_utility {
// Access pattern hint for a memory mapping (analogous to `madvise`).
enum class AccessPattern { None, Random, Sequential };
}  // namespace ad_utility

// The single platform seam: select the `FileMapping` primitive used by
// `MmapVector`. POSIX (`mmap`/`munmap`/`ftruncate`) is the default; Windows
// (Boost.Interprocess with a FILE_SHARE_DELETE handle) is the override. Both
// implementations share the same interface, so shared code (`MmapVectorImpl.h`)
// uses the neutral `ad_utility::FileMapping` and needs no `#ifdef _WIN32`. The
// enum above is defined before the include so the implementation's `advise`
// can refer to it.
#ifdef _WIN32
#include "util/windows/WindowsFileMapping.h"
namespace ad_utility {
using FileMapping = windows::FileMapping;
}
#else
#include "util/PosixFileMapping.h"
namespace ad_utility {
using FileMapping = posix::FileMapping;
}
#endif

#endif  // QLEVER_SRC_UTIL_FILEMAPPING_H
