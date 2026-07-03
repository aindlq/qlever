// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_DEFAULTSHAREDMUTEX_H
#define QLEVER_SRC_UTIL_DEFAULTSHAREDMUTEX_H

#include <shared_mutex>

// The single platform seam for the default shared mutex used by `Synchronized`,
// so shared code refers to the neutral `ad_utility::DefaultSharedMutex` with no
// `#ifdef _WIN32`. POSIX uses `std::shared_mutex` (the default). Windows must
// not: on MinGW, libstdc++'s `std::shared_mutex` wraps winpthreads'
// `pthread_rwlock`, which is lazily initialized on first `lock()` and races on
// cold concurrent first use (mingw-w64 bug #883); the losing thread gets
// `EINVAL`, which libstdc++ ignores in release builds, so it enters the
// "exclusive" section without the lock (a silent loss of mutual exclusion).
// Windows therefore uses `ThreadAgnosticSharedMutex`, built on a
// zero-initialized C++20 atomic (no lazy init, no race). See
// `util/windows/ThreadAgnosticSharedMutex.h` for the full rationale.
#ifdef _WIN32
#include "util/windows/ThreadAgnosticSharedMutex.h"
namespace ad_utility {
using DefaultSharedMutex = ThreadAgnosticSharedMutex;
}
#else
namespace ad_utility {
using DefaultSharedMutex = std::shared_mutex;
}
#endif

#endif  // QLEVER_SRC_UTIL_DEFAULTSHAREDMUTEX_H
