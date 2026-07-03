// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_PORTABLETIMEDWAIT_H
#define QLEVER_SRC_UTIL_PORTABLETIMEDWAIT_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

// Portable interruptible timed wait, keeping the single `#ifdef _WIN32` here so
// callers (e.g. `CancellationHandle.cpp`) stay platform-neutral.

namespace ad_utility {

// Wait up to `duration` for `predicate()` to become true, atomically releasing
// `lock` while waiting and holding it whenever `predicate` is evaluated.
// Returns `true` if the predicate became true (woke early), `false` on timeout.
// On POSIX this is a plain `cv.wait_for`. On Windows it polls the predicate
// with 1 ms sleeps instead: condition-variable timed waits there only have the
// system timer granularity (~16 ms), which is too coarse for callers with a
// short check interval; polling keeps the wake-up latency at most ~1 ms. `cv`
// is unused on Windows (the caller is woken by the predicate, not by a
// notification).
template <typename Rep, typename Period, typename Predicate>
bool waitForPredicateOrTimeout(
    std::condition_variable& cv, std::unique_lock<std::mutex>& lock,
    const std::chrono::duration<Rep, Period>& duration, Predicate predicate) {
#ifdef _WIN32
  (void)cv;
  auto deadline = std::chrono::steady_clock::now() + duration;
  while (std::chrono::steady_clock::now() < deadline) {
    if (predicate()) {
      return true;
    }
    lock.unlock();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    lock.lock();
  }
  return predicate();
#else
  return cv.wait_for(lock, duration, std::move(predicate));
#endif
}

}  // namespace ad_utility

#endif  // QLEVER_SRC_UTIL_PORTABLETIMEDWAIT_H
