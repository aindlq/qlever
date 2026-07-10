// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H
#define QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H

#include <shared_mutex>

// Platform seam for the default shared mutex used by `Synchronized`.
namespace ad_utility {
using DefaultSharedMutex = std::shared_mutex;
}

#endif  // QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H
