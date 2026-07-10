// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_PORTABLETIMEDWAIT_H
#define QLEVER_SRC_UTIL_SYS_PORTABLETIMEDWAIT_H

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <utility>

namespace ad_utility {

// Wait up to `duration` for `predicate()`, releasing `lock` while waiting.
// Returns `true` if the predicate became true, `false` on timeout.
template <typename Rep, typename Period, typename Predicate>
bool waitForPredicateOrTimeout(
    std::condition_variable& cv, std::unique_lock<std::mutex>& lock,
    const std::chrono::duration<Rep, Period>& duration, Predicate predicate) {
#ifdef _WIN32
  // Poll instead of `cv.wait_for`: Windows CV timed waits only have the ~16 ms
  // timer granularity, too coarse for short check intervals; polling caps the
  // latency at ~1 ms. `cv` is unused (the caller is woken by the predicate).
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

#endif  // QLEVER_SRC_UTIL_SYS_PORTABLETIMEDWAIT_H
