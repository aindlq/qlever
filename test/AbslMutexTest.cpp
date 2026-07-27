// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

// Tests for the Abseil mutex used as `Synchronized`'s default.
//
// Background: on MinGW, libstdc++'s `std::shared_mutex` is backed by
// winpthreads' `pthread_rwlock`, which is lazily initialized on the first
// `lock()` and races on cold concurrent first use (mingw-w64 bug #883,
// https://sourceforge.net/p/mingw-w64/bugs/883/): the thread that loses the
// init race gets `EINVAL`, libstdc++ ignores it in release builds, and that
// thread enters the "exclusive" section without holding the lock -> silent
// loss of mutual exclusion. The lazily allocated rwlock is also leaked when
// libstdc++ uses PTHREAD_RWLOCK_INITIALIZER (mingw-w64 bug #1012). QLever
// therefore uses `absl::Mutex` as its default on every platform.
//
// This file contains:
//  * GREEN regression guards that `absl::Mutex` and `Synchronized` preserve
//    mutual exclusion, also under concurrent first use.
//  * A DISABLED reproduction of the underlying platform bug on a raw
//    `std::shared_mutex`.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <shared_mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "util/Synchronized.h"

using ad_utility::Synchronized;

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
  for (int t = 0; t < numThreads; ++t) {
    threads.emplace_back(worker);
  }
  startBarrier.store(1, std::memory_order_release);
  for (auto& thread : threads) {
    thread.join();
  }
  return violations.load();
}

// Construct a FRESH mutex of type `Mutex` for each round and hit it
// cold-concurrently. The cold first use triggers winpthreads' lazy rwlock-init
// race. Returns total violations across all rounds.
template <typename Mutex>
long coldStartViolations(int rounds, int numThreads = 8, int iters = 4) {
  long total = 0;
  for (int r = 0; r < rounds; ++r) {
    Mutex mutex;
    std::atomic<int> barrier{0};
    total += exclusiveOverlapViolations(mutex, numThreads, iters, barrier);
  }
  return total;
}

}  // namespace

static_assert(ad_utility::AllowsLocking<absl::Mutex>::value);
static_assert(ad_utility::AllowsSharedLocking<absl::Mutex>::value);
static_assert(
    std::is_same_v<Synchronized<int>, Synchronized<int, absl::Mutex>>);

TEST(AbslMutex, ExclusiveLockProvidesMutualExclusion) {
  absl::Mutex mutex;
  long counter = 0;
  constexpr int kThreads = 8;
  constexpr int kIters = 50'000;
  auto worker = [&] {
    for (int i = 0; i < kIters; ++i) {
      std::lock_guard lock{mutex};
      ++counter;
    }
  };
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back(worker);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(counter, static_cast<long>(kThreads) * kIters);
}

TEST(AbslMutex, SharedLockAllowsConcurrentReaders) {
  absl::Mutex mutex;
  std::atomic<int> concurrent{0};
  std::atomic<int> maxConcurrent{0};
  constexpr int kReaders = 8;
  auto reader = [&] {
    std::shared_lock lock{mutex};
    int n = concurrent.fetch_add(1) + 1;
    int previous = maxConcurrent.load();
    while (n > previous && !maxConcurrent.compare_exchange_weak(previous, n)) {
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
    concurrent.fetch_sub(1);
  };
  std::vector<std::thread> threads;
  for (int t = 0; t < kReaders; ++t) {
    threads.emplace_back(reader);
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_GT(maxConcurrent.load(), 1);
}

TEST(AbslMutex, ExclusiveLockExcludesSharedLocks) {
  absl::Mutex mutex;
  std::atomic<bool> writerInside{false};
  std::atomic<long> violations{0};
  std::atomic<int> started{0};
  mutex.lock();
  writerInside.store(true);
  constexpr int kReaders = 8;
  std::vector<std::thread> threads;
  for (int t = 0; t < kReaders; ++t) {
    threads.emplace_back([&] {
      started.fetch_add(1);
      std::shared_lock lock{mutex};
      if (writerInside.load()) {
        violations.fetch_add(1);
      }
    });
  }
  while (started.load() < kReaders) {
  }
  std::this_thread::sleep_for(std::chrono::milliseconds{20});
  writerInside.store(false);
  mutex.unlock();
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(violations.load(), 0);
}

TEST(AbslMutex, ConcurrentFirstUseStress) {
  EXPECT_EQ(coldStartViolations<absl::Mutex>(1500), 0);
}

// Repeatedly exercise the lifecycle that leaks with MinGW's std::shared_mutex
// (#1012).
TEST(AbslMutex, RepeatedConstructLockDestroy) {
  for (size_t i = 0; i < 100'000; ++i) {
    absl::Mutex mutex;
    { std::lock_guard lock{mutex}; }
    { std::shared_lock lock{mutex}; }
  }
}

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
    Synchronized<Cell> synchronized;
    std::atomic<int> barrier{0};
    std::atomic<long> overlaps{0};
    auto worker = [&] {
      while (barrier.load(std::memory_order_acquire) == 0) {
      }
      for (int i = 0; i < kIters; ++i) {
        synchronized.withWriteLock([&](Cell& cell) {
          if (++cell.inside != 1) {
            overlaps.fetch_add(1, std::memory_order_relaxed);
          }
          ++cell.counter;
          for (int s = 0; s < 8; ++s) {
            gSink.fetch_add(1);
          }
          --cell.inside;
        });
      }
    };
    std::vector<std::thread> threads;
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back(worker);
    }
    barrier.store(1, std::memory_order_release);
    for (auto& thread : threads) {
      thread.join();
    }
    long got = synchronized.withWriteLock(
        [](const Cell& cell) { return cell.counter; });
    if (got != static_cast<long>(kThreads) * kIters || overlaps.load() != 0) {
      ++badRounds;
    }
  }
  EXPECT_EQ(badRounds, 0);
}

// Reproduces mingw-w64 bug #883 directly on a raw `std::shared_mutex` under
// cold concurrent first use. It passes on platforms without the bug.
TEST(SharedMutexColdStartRace, DISABLED_StdSharedMutexColdStartMingwBug883) {
  long violations = coldStartViolations<std::shared_mutex>(4000);
  EXPECT_EQ(violations, 0)
      << violations
      << " mutual-exclusion violations on a freshly constructed "
         "std::shared_mutex under cold concurrent first use";
}
