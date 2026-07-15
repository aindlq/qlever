// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSweepExecutor.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#endif

#include <atomic>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// For `vectorSearchThreadCap()` (the warm-up team size).
#include "services/vectorSearch/VectorIndex.h"

namespace qlever::vector {

namespace {

// One submitted sweep. `state_` is the caller<->worker handshake:
//   Queued    -> Running   (a worker claimed it)
//   Queued    -> Abandoned (the waiting caller was cancelled; never runs)
//   Running   -> Done      (finished; `error_` carries any exception)
//   Queued    -> Done      (failed at shutdown, `error_` set)
// The job is shared (`shared_ptr`) between the caller and the queue, so an
// abandoning caller can return while its entry is still queued.
struct SweepJob {
  enum State : int { Queued = 0, Running, Done, Abandoned };
  void (*fn_)(void*);
  void* arg_;
  std::exception_ptr error_;
  std::atomic<int> state_{Queued};
};

// Thread-local identity + scratch slot of an executor worker (null on every
// other thread).
struct WorkerScratch {
  std::vector<float> floats_;
  std::vector<int32_t> ints_;
};
thread_local WorkerScratch* tlWorkerScratch = nullptr;

// The live off-switch: QLEVER_VECTOR_SWEEP_MASTER=off|0|false|no. Read per
// call (not memoized) so a benchmark/test can flip it inside one process; a
// getenv is noise against a >= ms sweep.
bool sweepMasterEnabled() {
  const char* v = std::getenv("QLEVER_VECTOR_SWEEP_MASTER");
  if (v == nullptr) {
    return true;
  }
  std::string s{v};
  for (char& c : s) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  return !(s == "0" || s == "false" || s == "off" || s == "no");
}

// Pool size: QLEVER_VECTOR_SWEEP_EXECUTORS (default 1 -- the sweeps are
// memory-bandwidth-bound, so serializing them is near-optimal throughput;
// 2 lets an operator try concurrent sweeps). Clamped to [1, 16]. Read ONCE,
// at pool creation.
size_t sweepExecutorCount() {
  const char* v = std::getenv("QLEVER_VECTOR_SWEEP_EXECUTORS");
  if (v == nullptr || *v == '\0') {
    return 1;
  }
  char* end = nullptr;
  const long parsed = std::strtol(v, &end, 10);
  if (end == v || *end != '\0' || parsed < 1) {
    return 1;
  }
  return static_cast<size_t>(parsed > 16 ? 16 : parsed);
}

}  // namespace

struct SweepExecutor::Impl {
  std::mutex mutex_;
  // Workers sleep on `queueCv_`; callers wait for their job on `doneCv_`
  // (shared -- completions are rare events, a broadcast is fine).
  std::condition_variable queueCv_;
  std::condition_variable doneCv_;
  std::deque<std::shared_ptr<SweepJob>> queue_;
  bool shutdown_ = false;
  std::vector<std::thread> workers_;
  std::vector<std::unique_ptr<WorkerScratch>> scratch_;

  void workerMain(size_t workerIdx);
};

// ____________________________________________________________________________
void SweepExecutor::Impl::workerMain(size_t workerIdx) {
#if defined(__linux__)
  pthread_setname_np(pthread_self(), "VecSweepExec");
  // DEFINED placement: adopt the PROCESS's allowed-CPU set (the main
  // thread's mask -- the container/cpuset view), NOT whatever narrowed mask
  // the thread that happened to trigger pool creation had (OMP_PROC_BIND can
  // pin a master to a single core, and the OpenMP team this worker is about
  // to create INHERITS its creator's mask).
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(getpid(), sizeof(mask), &mask) == 0) {
    (void)sched_setaffinity(0, sizeof(mask), &mask);
  }
#endif
#ifdef _OPENMP
  // Warm the OpenMP team ONCE, at creation: team creation + thread placement
  // + stack first-touch happen here, deterministically at startup, instead
  // of inside the first query's timed sweep.
  {
    std::atomic<int> sink{0};
    const int warmThreads = vectorSearchThreadCap();
#pragma omp parallel num_threads(warmThreads)
    { sink.fetch_add(1, std::memory_order_relaxed); }
  }
#endif
  tlWorkerScratch = scratch_[workerIdx].get();
  std::unique_lock<std::mutex> lock{mutex_};
  while (true) {
    queueCv_.wait(lock, [&] { return shutdown_ || !queue_.empty(); });
    if (shutdown_) {
      // Fail any still-queued jobs so no caller hangs, then exit.
      while (!queue_.empty()) {
        auto job = std::move(queue_.front());
        queue_.pop_front();
        int expected = SweepJob::Queued;
        if (job->state_.compare_exchange_strong(expected, SweepJob::Running)) {
          job->error_ = std::make_exception_ptr(std::runtime_error(
              "The vector sweep executor was shut down while this sweep "
              "was still queued."));
          job->state_.store(SweepJob::Done, std::memory_order_release);
        }
      }
      doneCv_.notify_all();
      return;
    }
    auto job = std::move(queue_.front());
    queue_.pop_front();
    lock.unlock();
    int expected = SweepJob::Queued;
    if (job->state_.compare_exchange_strong(expected, SweepJob::Running)) {
      try {
        job->fn_(job->arg_);
      } catch (...) {
        job->error_ = std::current_exception();
      }
      // Publish completion under the mutex (no missed wakeup) + broadcast.
      lock.lock();
      job->state_.store(SweepJob::Done, std::memory_order_release);
      doneCv_.notify_all();
    } else {
      lock.lock();  // abandoned while queued -- skip it
    }
  }
}

// ____________________________________________________________________________
SweepExecutor::SweepExecutor() : impl_{std::make_unique<Impl>()} {
  const size_t count = sweepExecutorCount();
  impl_->scratch_.reserve(count);
  impl_->workers_.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    impl_->scratch_.push_back(std::make_unique<WorkerScratch>());
  }
  for (size_t i = 0; i < count; ++i) {
    impl_->workers_.emplace_back([this, i] { impl_->workerMain(i); });
  }
}

// ____________________________________________________________________________
SweepExecutor::~SweepExecutor() {
  {
    std::lock_guard<std::mutex> lock{impl_->mutex_};
    impl_->shutdown_ = true;
  }
  impl_->queueCv_.notify_all();
  for (auto& worker : impl_->workers_) {
    worker.join();
  }
}

// ____________________________________________________________________________
SweepExecutor& SweepExecutor::instance() {
  // A function-local static with JOINABLE workers: the destructor signals and
  // joins at static destruction, so no worker outlives the object (unlike a
  // detached thread over a Meyers static, which use-after-frees at exit).
  static SweepExecutor executor;
  return executor;
}

// ____________________________________________________________________________
void SweepExecutor::ensureStarted() {
  if (sweepMasterEnabled()) {
    (void)instance();
  }
}

// ____________________________________________________________________________
bool SweepExecutor::onExecutorThread() { return tlWorkerScratch != nullptr; }

// ____________________________________________________________________________
size_t SweepExecutor::numWorkers() const { return impl_->workers_.size(); }

// ____________________________________________________________________________
bool SweepExecutor::shouldRoute() {
  return sweepMasterEnabled() && !onExecutorThread();
}

// ____________________________________________________________________________
float* SweepExecutor::floatScratch(size_t n) {
  WorkerScratch* s = tlWorkerScratch;
  if (s == nullptr) {
    return nullptr;
  }
  if (s->floats_.size() < n) {
    s->floats_.resize(n);
  }
  return s->floats_.data();
}

// ____________________________________________________________________________
int32_t* SweepExecutor::int32Scratch(size_t n) {
  WorkerScratch* s = tlWorkerScratch;
  if (s == nullptr) {
    return nullptr;
  }
  if (s->ints_.size() < n) {
    s->ints_.resize(n);
  }
  return s->ints_.data();
}

// ____________________________________________________________________________
void SweepExecutor::run(void (*fn)(void*), void* arg,
                        const std::function<void()>& checkInterrupt) {
  auto job = std::make_shared<SweepJob>();
  job->fn_ = fn;
  job->arg_ = arg;
  {
    std::lock_guard<std::mutex> lock{impl_->mutex_};
    if (impl_->shutdown_) {
      // Static destruction has begun (no server queries at this point); run
      // inline rather than hanging on a dead queue.
      fn(arg);
      return;
    }
    impl_->queue_.push_back(job);
  }
  impl_->queueCv_.notify_one();

  std::unique_lock<std::mutex> lock{impl_->mutex_};
  while (true) {
    const int state = job->state_.load(std::memory_order_acquire);
    if (state == SweepJob::Done) {
      break;
    }
    // The completion broadcast wakes us immediately; the timeout exists only
    // to poll the caller's cancellation while the job is still QUEUED.
    impl_->doneCv_.wait_for(lock, std::chrono::milliseconds(2));
    if (checkInterrupt &&
        job->state_.load(std::memory_order_acquire) == SweepJob::Queued) {
      try {
        checkInterrupt();
      } catch (...) {
        int expected = SweepJob::Queued;
        if (job->state_.compare_exchange_strong(expected,
                                                SweepJob::Abandoned)) {
          // Abandoned before a worker claimed it: the task never runs, so
          // returning (and destroying everything it captured) is safe. The
          // queue's shared_ptr keeps the job block alive for the worker's
          // eventual skip.
          throw;
        }
        // A worker claimed it concurrently -- it is running over our stack
        // state, so we MUST wait for Done (the task polls the same callback
        // per chunk and will finish promptly).
      }
    }
  }
  if (job->error_) {
    std::rethrow_exception(job->error_);
  }
}

}  // namespace qlever::vector
