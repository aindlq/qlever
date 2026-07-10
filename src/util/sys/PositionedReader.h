// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H

// Platform seam for `File`'s positioned reads.
#include "util/sys/PosixPositionedReader.h"

namespace ad_utility {
using PositionedReader = posix::PositionedReader;
}

#endif  // QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H
