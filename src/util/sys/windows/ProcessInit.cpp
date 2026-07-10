// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

// Process-wide runtime fixes for the Windows build. This file is compiled
// into the `qleverWindowsInit` object library, which is linked into every
// QLever executable (see `CMakeLists.txt` in this directory).

#include <fcntl.h>
#include <stdio.h>

#include <chrono>
#include <ctime>

#include "util/sys/windows/WindowsUndefs.h"

// Make binary mode the default for ALL CRT file opens in the process
// (matching Linux semantics). Several dependencies (e.g. libspatialjoin)
// call open()/fopen() without O_BINARY/"b" on raw binary data; Windows'
// text-mode default then inserts/strips CR bytes and treats 0x1A as EOF,
// silently corrupting binary files.
namespace {
struct QleverProcessInit {
  QleverProcessInit() {
    _set_fmode(_O_BINARY);
    // Raise the C runtime's open-stream limit from the default 512 to the
    // maximum 8192. Building a large index merges the per-batch partial
    // vocabularies by `fopen`ing them all at once (two streams per batch);
    // for billion-triple datasets the batch count exceeds 512, which would
    // otherwise fail mid-merge with `! ERROR opening file
    // "<...>.partial-vocab.words.tmp.N" ... (Input/output error)`. POSIX has
    // no equivalent low default (it is governed by `RLIMIT_NOFILE`).
    _setmaxstdio(8192);
  }
};
QleverProcessInit qleverProcessInit;
}  // namespace

// Precise sleep, interposed via `ld --wrap=nanosleep64`.
//
// The default Windows timer tick is ~15.6 ms and neither `timeBeginPeriod`
// nor the global timer-resolution policy reliably restore 1 ms sleeps (e.g.
// in virtualized or session-0 environments). libstdc++'s
// `std::this_thread::sleep_for` compiles to a call to winpthreads'
// `nanosleep64`, which we replace here: a high-resolution waitable timer (a
// hardware deadline timer, unaffected by the system tick) does the bulk of the
// wait, then we spin ONLY the sub-millisecond residual. The residual spin is
// not for extra precision but for correctness: the waitable timer fires on a
// different clock than the `steady_clock` callers measure sleeps with, and can
// come up a few microseconds short by that clock; POSIX `nanosleep` never
// returns early, and QLever code (and tests) rely on `elapsed >= requested`.
// This is a microsecond tail, not the multi-millisecond busy-wait a naive
// implementation would use.
extern "C" int __wrap_nanosleep64(const struct _timespec64* req,
                                  struct _timespec64* rem) {
  (void)rem;
  // Saturate to a large-but-safe bound so a "sleep (almost) forever" idiom
  // (`sleep_for(duration::max())`) can't overflow the multiplication, the
  // `deadline` below, or the fallback `Sleep()` DWORD - any of which would wrap
  // and turn the sleep into a busy-spin. Nothing in QLever sleeps for days.
  constexpr long long kMaxNs = 30LL * 24 * 3600 * 1000000000LL;  // ~30-day cap
  long long totalNs;
  if (req->tv_sec > kMaxNs / 1000000000LL) {
    totalNs = kMaxNs;
  } else {
    totalNs = static_cast<long long>(req->tv_sec) * 1000000000LL + req->tv_nsec;
    if (totalNs > kMaxNs) {
      totalNs = kMaxNs;
    }
  }
  if (totalNs <= 0) {
    return 0;
  }
  using Clock = std::chrono::steady_clock;
  const auto deadline = Clock::now() + std::chrono::nanoseconds(totalNs);
  // A per-thread high-resolution waitable timer, released when the thread
  // exits. (An unwrapped `static thread_local HANDLE` would leak one timer
  // object per thread that ever sleeps, for the whole process lifetime.)
  struct TimerHandle {
    HANDLE handle = CreateWaitableTimerExW(
        nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION,
        TIMER_ALL_ACCESS);
    ~TimerHandle() {
      if (handle != nullptr) {
        CloseHandle(handle);
      }
    }
  };
  static thread_local TimerHandle timer;
  bool waited = false;
  if (timer.handle != nullptr) {
    LARGE_INTEGER dueTime;
    // Negative = relative time, in 100 ns units.
    dueTime.QuadPart = -(totalNs / 100);
    // Only wait on the timer if it was actually armed; otherwise fall through
    // to `Sleep` rather than blocking forever on a never-signaled timer.
    if (SetWaitableTimer(timer.handle, &dueTime, 0, nullptr, nullptr, FALSE)) {
      WaitForSingleObject(timer.handle, INFINITE);
      waited = true;
    }
  }
  if (!waited) {
    // Fall back to millisecond-granularity `Sleep` if a high-resolution timer
    // is unavailable or could not be armed (it rounds up, matching the coarse
    // default tick anyway).
    Sleep(static_cast<DWORD>((totalNs + 999999) / 1000000));
  }
  // Spin only the residual so `elapsed >= requested` holds against the caller's
  // `steady_clock` (usually a no-op or a few microseconds).
  while (Clock::now() < deadline) {
    YieldProcessor();
  }
  return 0;
}
