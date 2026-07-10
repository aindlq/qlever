// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_EAGERINITIALIZEDSHAREDMUTEX_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_EAGERINITIALIZEDSHAREDMUTEX_H

#include <shared_mutex>

namespace ad_utility {

// A thin `std::shared_mutex` wrapper that eagerly initializes MinGW's
// underlying winpthreads rwlock. It is used as the default mutex of
// `Synchronized` on Windows (see `util/Synchronized.h`).
//
// WHY: libstdc++'s `std::shared_mutex` is backed by winpthreads'
// `pthread_rwlock_t`, which is just an `intptr_t` set to a static-init
// sentinel and lazily turned into a real lock on the first `lock()` via
// double-checked locking. Under cold concurrent first use, the thread that
// loses that init race receives `EINVAL` (mingw-w64 bug #883,
// https://sourceforge.net/p/mingw-w64/bugs/883/), and libstdc++'s
// `__shared_mutex_pthread::lock()` does not handle `EINVAL` in release builds
// (`__glibcxx_assert` is a no-op), so it returns as if the lock were held.
// Publicly this race is usually reported as a deadlock; here we reproduced and
// verified it as a silent loss of mutual exclusion: two threads enter the
// "exclusive" section of a freshly constructed `Synchronized<File>` at once,
// observed as interleaved writers corrupting index files. The bug is
// rwlock-specific:
// winpthreads resolves the same lazy-init race correctly for `std::mutex`
// (a CAS in mutex.c) and `std::condition_variable` (cond.c), so only
// `std::shared_mutex`/`shared_timed_mutex` are affected.
//
// This is NOT about releasing a lock from a non-owning thread: QLever's
// writers lock and unlock on the same thread (verified empirically: zero
// cross-thread unlocks in the failing tests). The cure is simply to force the
// lazy initialization to happen in this wrapper's constructor, before the
// mutex can be published to other threads. All operations after construction
// delegate directly to `std::shared_mutex`.
//
// NOTE: This covers QLever's own `Synchronized` usage only. Bare
// `std::shared_mutex` in code we do not control (e.g. the ANTLR4 C++ runtime,
// which switched to it in antlr/antlr4#3335) has the same exposure on MinGW
// and is not covered here. The common real-world workaround is to avoid
// `std::shared_mutex` (Ceph's Windows port switched to `boost::shared_mutex`);
// a global fix would need a patched winpthreads or the `mcf` thread model.
//
// NOTE: The implementation is portable C++ and compiles on any platform;
// it lives in `util/sys/windows/` because only the Windows build needs it.
class EagerInitializedSharedMutex {
 private:
  std::shared_mutex mutex_;

 public:
  EagerInitializedSharedMutex() {
    mutex_.lock();
    mutex_.unlock();
  }

  void lock() { mutex_.lock(); }
  bool try_lock() { return mutex_.try_lock(); }
  void unlock() { mutex_.unlock(); }

  void lock_shared() { mutex_.lock_shared(); }
  bool try_lock_shared() { return mutex_.try_lock_shared(); }
  void unlock_shared() { mutex_.unlock_shared(); }
};

}  // namespace ad_utility

#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_EAGERINITIALIZEDSHAREDMUTEX_H
