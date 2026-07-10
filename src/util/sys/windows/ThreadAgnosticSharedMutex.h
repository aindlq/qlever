// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_THREADAGNOSTICSHAREDMUTEX_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_THREADAGNOSTICSHAREDMUTEX_H

#include <atomic>
#include <cstdint>

namespace ad_utility {

// A shared mutex built on a zero-initialized C++20 atomic. It is used as the
// default mutex of `Synchronized` on Windows (see `util/Synchronized.h`).
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
// cross-thread unlocks in the failing tests). The cure is simply an
// always-initialized primitive. `std::atomic<int64_t> state_{0}` is
// zero-initialized at construction with no lazy first-use step, so the race
// cannot occur. (Semantically this is also a counting rwlock with no
// ownership affinity, like `std::counting_semaphore`, but that property is
// incidental here, not the reason it is needed.)
//
// NOTE: This covers QLever's own `Synchronized` usage only. Bare
// `std::shared_mutex` in code we do not control (e.g. the ANTLR4 C++ runtime,
// which switched to it in antlr/antlr4#3335) has the same exposure on MinGW
// and is not covered here. The common real-world workaround is to avoid
// `std::shared_mutex` (Ceph's Windows port switched to `boost::shared_mutex`);
// a global fix would need a patched winpthreads or the `mcf` thread model.
//
// NOTE: The implementation is portable C++20 and compiles on any platform;
// it lives in `util/sys/windows/` because only the Windows build needs it.
class ThreadAgnosticSharedMutex {
 private:
  // 0 = free, -1 = exclusively locked, > 0 = number of shared holders.
  std::atomic<int64_t> state_{0};

 public:
  // NOTE: reader-preferring - a continuous stream of shared holders can starve
  // an exclusive `lock()` (the same default as glibc's `std::shared_mutex`, so
  // this is parity, not a regression). `compare_exchange_strong` (not `_weak`)
  // is required here: a spurious weak failure would leave `expected == 0` and
  // then `wait(0)` would block on a *free* mutex until some unrelated thread
  // cycled it. Impossible on x86-64 (weak == strong there), but real on ARM64 -
  // and this class is exercised on the macOS/ARM64 CI.
  void lock() {
    int64_t expected = 0;
    while (!state_.compare_exchange_strong(
        expected, -1, std::memory_order_acquire, std::memory_order_relaxed)) {
      state_.wait(expected, std::memory_order_relaxed);
      expected = 0;
    }
  }
  void unlock() {
    state_.store(0, std::memory_order_release);
    state_.notify_all();
  }
  void lock_shared() {
    for (;;) {
      int64_t current = state_.load(std::memory_order_relaxed);
      if (current >= 0) {
        if (state_.compare_exchange_weak(current, current + 1,
                                         std::memory_order_acquire,
                                         std::memory_order_relaxed)) {
          return;
        }
      } else {
        state_.wait(current, std::memory_order_relaxed);
      }
    }
  }
  void unlock_shared() {
    if (state_.fetch_sub(1, std::memory_order_release) == 1) {
      state_.notify_all();
    }
  }
};

}  // namespace ad_utility

#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_THREADAGNOSTICSHAREDMUTEX_H
