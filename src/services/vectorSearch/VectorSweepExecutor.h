// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSWEEPEXECUTOR_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSWEEPEXECUTOR_H

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <type_traits>

namespace qlever::vector {

// A PERSISTENT executor for the whole-index vector sweeps.
//
// WHY: the sweeps enter their `#pragma omp parallel` regions from the CALLING
// thread, and libgomp keeps one worker team PER MASTER THREAD. A server
// executes every query on a rotating asio pool thread, so each query becomes
// a brand-new OpenMP master: it pays worker-team creation, first-touch of the
// team's stacks/ICVs, and gets whatever thread placement the scheduler
// improvises -- measured 27-29 ms per production i8 sweep from a fresh
// master vs 19 ms from a reused one. Funnelling every sweep through ONE
// long-lived worker thread (created once, with a defined CPU placement and a
// pre-warmed team) makes every query run on the same warm, well-placed team.
//
// STRUCTURE: `instance()` owns a small pool of JOINABLE worker threads
// (default 1 -- the sweeps are memory-bandwidth-bound, so running two
// concurrently mostly slows both; `QLEVER_VECTOR_SWEEP_EXECUTORS` overrides
// for A/B). Each worker, at creation:
//   * pins itself to the PROCESS's allowed-CPU set (`sched_getaffinity` of
//     the process, applied via `sched_setaffinity` on itself), so its team's
//     placement does not depend on which thread happened to create the pool;
//   * spins its OpenMP team up once with an empty parallel region, so team
//     creation/placement happens at startup, not inside the first query.
// The pool is created on FIRST USE (a thread-safe function-local static) and
// shut down deterministically at static destruction: the destructor signals,
// fails any still-queued jobs, and JOINS every worker -- no detached thread
// can outlive the object.
//
// SUBMISSION (`run`): the caller blocks until its task finished on a worker
// and rethrows the task's exception. While the task is still QUEUED behind
// another sweep, the caller polls its `checkInterrupt` callback -- a
// cancelled caller atomically ABANDONS the queued task (a worker will skip
// it; it never starts) and returns promptly. Once the task RUNS, the caller
// waits for it to finish -- the sweep bodies poll the same callback per
// chunk, so a cancelled running task also returns quickly.
//
// REENTRANCY: a task that is already running ON an executor worker must
// never re-queue (executors=1 would deadlock); `runVectorSweep` detects
// "am I on the executor?" via a thread-local and runs nested sweeps INLINE
// on the same (warm) master. CONTRACT: never call `runVectorSweep` from
// INSIDE an OpenMP parallel region (a team WORKER is not the executor
// thread, so it would re-queue behind its own master and deadlock) -- all
// funnel sites wrap whole regions, never code within one.
//
// OFF-SWITCH: `QLEVER_VECTOR_SWEEP_MASTER=off|0|false|no` routes every sweep
// inline on the calling thread (the pre-executor behaviour). Read LIVE (one
// getenv per sweep -- noise against a >= ms sweep) so a benchmark can A/B
// executor-on vs -off in one process.
class SweepExecutor {
 public:
  // The one process-wide instance. Created on first call; its workers are
  // joined when static destructors run.
  static SweepExecutor& instance();

  // Create (and warm) the pool now if routing is enabled -- called from
  // `VectorIndex::open` so the team exists, placed, before the first query.
  static void ensureStarted();

  // True iff sweeps should be handed to the executor by the CALLING thread:
  // routing not disabled via env AND the caller is not itself an executor
  // worker (nested sweeps run inline).
  static bool shouldRoute();

  // True iff the calling thread is one of the executor's workers.
  static bool onExecutorThread();

  // Per-WORKER reusable scratch (grown once, reused across sweeps -- kills
  // the per-query multi-MB `new float[n]` + first-touch churn). Returns the
  // calling WORKER's buffer of >= n elements, or nullptr when the caller is
  // not an executor worker (the caller then allocates locally). CONTRACT: at
  // most ONE live use of each buffer per task -- only the top-level task
  // body may take it (nested helpers must not).
  static float* floatScratch(size_t n);
  static int32_t* int32Scratch(size_t n);

  // Run `fn(arg)` on a worker; block until done; rethrow its exception.
  // `checkInterrupt` (may be empty) is polled while the task is QUEUED so a
  // cancelled caller can abandon the wait (see above).
  void run(void (*fn)(void*), void* arg,
           const std::function<void()>& checkInterrupt);

  // The pool size (`QLEVER_VECTOR_SWEEP_EXECUTORS`, default 1).
  size_t numWorkers() const;

  ~SweepExecutor();
  SweepExecutor(const SweepExecutor&) = delete;
  SweepExecutor& operator=(const SweepExecutor&) = delete;

 private:
  SweepExecutor();
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// THE chokepoint: every whole-index sweep's OpenMP parallel region is entered
// through this function, so all sweep routes (i8/bf16 block, per-row, CSLS
// contiguous + gather, histogram select, ...) run on the persistent executor
// by construction. Runs `task()` inline when routing is off or the caller is
// already an executor worker.
template <typename F>
inline void runVectorSweep(const std::function<void()>& checkInterrupt,
                           F&& task) {
  if (!SweepExecutor::shouldRoute()) {
    task();
    return;
  }
  using Fn = std::remove_reference_t<F>;
  SweepExecutor::instance().run(
      [](void* p) { (*static_cast<Fn*>(p))(); },
      const_cast<void*>(static_cast<const void*>(std::addressof(task))),
      checkInterrupt);
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSWEEPEXECUTOR_H
