// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem Kozlov <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_UTIL_SYS_FILEMAPPING_H
#define QLEVER_SRC_UTIL_SYS_FILEMAPPING_H

// Platform seam for `MmapVector`'s file mapping.
#include "util/sys/AccessPattern.h"
#include "util/sys/PosixFileMapping.h"

namespace ad_utility {
using FileMapping = posix::FileMapping;
}

#endif  // QLEVER_SRC_UTIL_SYS_FILEMAPPING_H
