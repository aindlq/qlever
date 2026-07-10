// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem Kozlov <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H
#define QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H

// Platform seam for `File`'s positioned reads.
#include "util/sys/PosixPositionedReader.h"

namespace ad_utility {
using PositionedReader = posix::PositionedReader;
}

#endif  // QLEVER_SRC_UTIL_SYS_POSITIONEDREADER_H
