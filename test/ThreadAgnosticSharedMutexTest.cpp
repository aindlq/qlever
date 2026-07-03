// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

// Tests for the synchronization fix discovered during the Windows port.
//
// Background: on MinGW, libstdc++'s `std::shared_mutex` is backed by
// winpthreads' `pthread_rwlock`, which is lazily initialized on the first
// `lock()` and races on cold concurrent first use (mingw-w64 bug #883,
// https://sourceforge.net/p/mingw-w64/bugs/883/): the thread that loses the
// init race gets `EINVAL`, libstdc++ ignores it in release builds, and that
// thread enters the "exclusive" section without holding the lock -> silent
// loss of mutual exclusion. QLever's `Synchronized` therefore uses
// `ThreadAgnosticSharedMutex` (a zero-initialized atomic, no lazy init) on
// Windows.
//
// This file contains:
//  * GREEN regression guards (run everywhere) that `ThreadAgnosticSharedMutex`
//    and the `Synchronized` *default* mutex preserve mutual exclusion, also
//    under cold concurrent first use. These would turn RED on Windows if the
//    default were reverted to `std::shared_mutex`.
//  * A DISABLED reproduction of the underlying platform bug on a raw
//    `std::shared_mutex` -- which is exactly the primitive the ANTLR4 C++
//    runtime uses internally -- so it also demonstrates ANTLR's exposure.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <shared_mutex>
#include <thread>
#include <vector>

#include "util/Synchronized.h"
#include "util/windows/ThreadAgnosticSharedMutex.h"

using ad_utility::Synchronized;
using ad_utility::ThreadAgnosticSharedMutex;

namespace {

// A never-optimized-away sink, used to widen the in-critical-section window so
// a momentary loss of mutual exclusion is actually observed.
std::atomic<long> gSink{0};

// `numThreads` threads each take the EXCLUSIVE lock `iters` times after a
// common start barrier. A non-negative `inside` counter (atomic only so the
// detection itself is race-free) must never exceed 1 while the lock is held;
// if it does, two threads are in the "exclusive" section at once. Returns the
// number of such mutual-exclusion violations.
template <typename ExclusiveLockable>
long exclusiveOverlapViolations(ExclusiveLockable& m, int numThreads, int iters,
                                std::atomic<int>& startBarrier) {
  std::atomic<long> violations{0};
  std::atomic<int> inside{0};
  auto worker = [&] {
    while (startBarrier.load(std::memory_order_acquire) == 0) {
    }  // spin until released, so first locks are concurrent
    for (int i = 0; i < iters; ++i) {
      m.lock();
      if (inside.fetch_add(1, std::memory_order_acq_rel) != 0) {
        violations.fetch_add(1, std::memory_order_relaxed);
      }
      for (int s = 0; s < 16; ++s) {
        gSink.fetch_add(1, std::memory_order_relaxed);  // widen the window
      }
      inside.fetch_sub(1, std::memory_order_acq_rel);
      m.unlock();
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(numThreads);
  for (int t = 0; t < numThreads; ++t) threads.emplace_back(worker);
  startBarrier.store(1, std::memory_order_release);  // release all at once
  for (auto& th : threads) th.join();
  return violations.load();
}

// Construct a FRESH mutex of type `Mutex` for each of `rounds` rounds and hit
// it cold-concurrently. The cold first use is what triggers winpthreads' lazy
// rwlock-init race. Returns total violations across all rounds.
template <typename Mutex>
long coldStartViolations(int rounds, int numThreads = 8, int iters = 4) {
  long total = 0;
  for (int r = 0; r < rounds; ++r) {
    Mutex m;  // fresh; on winpthreads the real rwlock is built on first lock()
    std::atomic<int> barrier{0};
    total += exclusiveOverlapViolations(m, numThreads, iters, barrier);
  }
  return total;
}

}  // namespace

// ---------------------------------------------------------------------------
// ThreadAgnosticSharedMutex: basic correctness (green on every platform)
// ---------------------------------------------------------------------------

TEST(ThreadAgnosticSharedMutex, ExclusiveLockProvidesMutualExclusion) {
  ThreadAgnosticSharedMutex m;
  long counter = 0;  // non-atomic: lost updates iff mutual exclusion fails
  constexpr int kThreads = 8;
  constexpr int kIters = 50000;
  auto worker = [&] {
    for (int i = 0; i < kIters; ++i) {
      m.lock();
      ++counter;
      m.unlock();
    }
  };
  std::vector<std::thread> ts;
  for (int t = 0; t < kThreads; ++t) ts.emplace_back(worker);
  for (auto& t : ts) t.join();
  EXPECT_EQ(counter, static_cast<long>(kThreads) * kIters);
}

TEST(ThreadAgnosticSharedMutex, SharedLockAllowsConcurrentReaders) {
  ThreadAgnosticSharedMutex m;
  std::atomic<int> concurrent{0};
  std::atomic<int> maxConcurrent{0};
  constexpr int kReaders = 8;
  auto reader = [&] {
    m.lock_shared();
    int n = concurrent.fetch_add(1) + 1;
    int prev = maxConcurrent.load();
    while (n > prev && !maxConcurrent.compare_exchange_weak(prev, n)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(20));  // overlap
    concurrent.fetch_sub(1);
    m.unlock_shared();
  };
  std::vector<std::thread> ts;
  for (int t = 0; t < kReaders; ++t) ts.emplace_back(reader);
  for (auto& t : ts) t.join();
  EXPECT_GT(maxConcurrent.load(), 1);  // readers genuinely shared the lock
}

TEST(ThreadAgnosticSharedMutex, ExclusiveLockExcludesSharedLocks) {
  ThreadAgnosticSharedMutex m;
  std::atomic<bool> writerInside{false};
  std::atomic<long> violations{0};
  std::atomic<int> started{0};
  m.lock();  // main thread holds the exclusive lock
  writerInside.store(true);
  constexpr int kReaders = 8;
  std::vector<std::thread> ts;
  for (int t = 0; t < kReaders; ++t) {
    ts.emplace_back([&] {
      started.fetch_add(1);
      m.lock_shared();
      if (writerInside.load()) violations.fetch_add(1);
      m.unlock_shared();
    });
  }
  while (started.load() < kReaders) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  writerInside.store(false);
  m.unlock();  // release; readers may now proceed
  for (auto& t : ts) t.join();
  EXPECT_EQ(violations.load(), 0);
}

// ---------------------------------------------------------------------------
// Cold-start mutual exclusion: the regression guards (green on every platform)
// ---------------------------------------------------------------------------

// `ThreadAgnosticSharedMutex` is zero-initialized, so it has no lazy first-use
// init and the winpthreads race cannot occur. This must hold everywhere.
TEST(ThreadAgnosticSharedMutex, ColdStartMutualExclusionStress) {
  EXPECT_EQ(coldStartViolations<ThreadAgnosticSharedMutex>(1500), 0);
}

// The real-world guard, at the level QLever actually uses: a freshly
// constructed `Synchronized` locked from several threads at once (e.g.
// `CompressedRelationWriter::outfile_`). With the Windows default reverted to
// `std::shared_mutex` this loses updates; with `ThreadAgnosticSharedMutex` it
// must not. Detects both overlap (sentinel) and lost updates (counter).
TEST(Synchronized, ColdStartMutualExclusionWithDefaultMutex) {
  struct Cell {
    long counter = 0;
    int inside = 0;
  };
  constexpr int kRounds = 1500;
  constexpr int kThreads = 8;
  constexpr int kIters = 8;
  long badRounds = 0;
  for (int r = 0; r < kRounds; ++r) {
    Synchronized<Cell> sync;  // fresh -> fresh default mutex -> cold first use
    std::atomic<int> barrier{0};
    std::atomic<long> overlaps{0};
    auto worker = [&] {
      while (barrier.load(std::memory_order_acquire) == 0) {
      }
      for (int i = 0; i < kIters; ++i) {
        sync.withWriteLock([&](Cell& c) {
          if (++c.inside != 1) overlaps.fetch_add(1, std::memory_order_relaxed);
          ++c.counter;
          for (int s = 0; s < 8; ++s) gSink.fetch_add(1);
          --c.inside;
        });
      }
    };
    std::vector<std::thread> ts;
    for (int t = 0; t < kThreads; ++t) ts.emplace_back(worker);
    barrier.store(1, std::memory_order_release);
    for (auto& t : ts) t.join();
    long got = sync.withWriteLock([](const Cell& c) { return c.counter; });
    if (got != static_cast<long>(kThreads) * kIters || overlaps.load() != 0) {
      ++badRounds;
    }
  }
  EXPECT_EQ(badRounds, 0);
}

// ---------------------------------------------------------------------------
// Reproduction of the underlying platform bug (DISABLED: fails ONLY on MinGW)
// ---------------------------------------------------------------------------

// Reproduces mingw-w64 bug #883 directly on a raw `std::shared_mutex` under
// cold concurrent first use. This is exactly the primitive the ANTLR4 C++
// runtime uses internally (`runtime/src/internal/Synchronization.h`'s
// `SharedMutex`, and `Parser.cpp`'s `BypassAltsAtnCache`), so a non-zero
// result here is also a demonstration of ANTLR's exposure -- which QLever's
// `ThreadAgnosticSharedMutex` does NOT cover.
//
// DISABLED because it passes on glibc (no bug) and fails only on MinGW. To
// reproduce on Windows:
//   ./ThreadAgnosticSharedMutexTest --gtest_also_run_disabled_tests
//       --gtest_filter='*StdSharedMutexColdStart*'
TEST(SharedMutexColdStartRace, DISABLED_StdSharedMutexColdStartMingwBug883) {
  long violations = coldStartViolations<std::shared_mutex>(4000);
  EXPECT_EQ(violations, 0)
      << violations
      << " mutual-exclusion violations on a freshly-constructed "
         "std::shared_mutex under cold concurrent first use. On MinGW this is "
         "winpthreads bug #883; the ANTLR4 runtime's std::shared_mutex has the "
         "same exposure and is not covered by ThreadAgnosticSharedMutex.";
}
