// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_POSITIONEDREADER_H
#define QLEVER_SRC_UTIL_POSITIONEDREADER_H

// The single platform seam: select the `PositionedReader` used by `File` for
// positioned reads (`read` at an explicit offset, POSIX `pread` semantics) with
// an optional memory-mapped fast path. POSIX (`pread`, plus an opt-in
// `QLEVER_LINUX_MMAP` mmap) is the default; Windows (a `pread`-emulating handle
// pool plus an always-on-when-enabled mmap) is the override. Both share the
// same interface (`readAtOffset`, `enableMemoryMappedReads`, `close`, move
// semantics), so `File.h` uses the neutral `ad_utility::PositionedReader` and
// needs no `#ifdef _WIN32`.
#ifdef _WIN32
#include "util/windows/WindowsPositionedReader.h"
namespace ad_utility {
using PositionedReader = windows::PositionedReader;
}
#else
#include "util/PosixPositionedReader.h"
namespace ad_utility {
using PositionedReader = posix::PositionedReader;
}
#endif

#endif  // QLEVER_SRC_UTIL_POSITIONEDREADER_H
