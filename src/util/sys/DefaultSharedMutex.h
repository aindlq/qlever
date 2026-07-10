// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H
#define QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H

#include <shared_mutex>

// Platform seam for the default shared mutex used by `Synchronized`.
#ifdef _WIN32
// Not `std::shared_mutex`: on MinGW it wraps winpthreads' `pthread_rwlock`,
// whose lazy init races on cold concurrent first use (mingw-w64 #883) and can
// silently drop mutual exclusion. See
// `util/sys/windows/ThreadAgnosticSharedMutex.h`.
#include "util/sys/windows/ThreadAgnosticSharedMutex.h"
namespace ad_utility {
using DefaultSharedMutex = ThreadAgnosticSharedMutex;
}
#else
namespace ad_utility {
using DefaultSharedMutex = std::shared_mutex;
}
#endif

#endif  // QLEVER_SRC_UTIL_SYS_DEFAULTSHAREDMUTEX_H
