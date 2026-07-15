// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests of the persistent vector-sweep executor (`SweepExecutor` /
// `runVectorSweep`):
//   * result identity: a plain whole-index two-layer i8 top-k (both the
//     blocked-heap and the histogram-select routes, plus the fine bf16
//     sweep) returns IDENTICAL results with the executor on vs off;
//   * exception propagation from the task to the caller;
//   * reentrancy: a task already running ON the executor runs nested sweeps
//     inline (no self-deadlock at pool size 1);
//   * cancellation: a caller whose task is still QUEUED behind another sweep
//     abandons the wait promptly, and the abandoned task never runs;
//   * per-worker scratch: available on the executor, absent off it.

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorSweepExecutor.h"

namespace {
using namespace qlever::vector;

// RAII: set QLEVER_VECTOR_SWEEP_MASTER for one test, restore the default
// (unset = on) afterwards. The routing flag is read live, so this flips the
// executor on/off between searches within one process.
struct SweepMasterEnv {
  explicit SweepMasterEnv(const char* value) {
    setenv("QLEVER_VECTOR_SWEEP_MASTER", value, 1);
  }
  ~SweepMasterEnv() { unsetenv("QLEVER_VECTOR_SWEEP_MASTER"); }
};

Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

// One shared synthetic two-layer i8 (coarse) + bf16 (fine) index, above the
// parallel threshold so the OpenMP sweep regions actually engage. Same shape
// as the `VectorI8ScanTest` fixture, separate directory.
struct ExecutorFixture {
  static constexpr size_t kN = 3000;
  static constexpr size_t kDim = 100;
  VectorIndex index;
  std::vector<float> query;

  ExecutorFixture() {
    const std::string dir =
        (std::filesystem::temp_directory_path() / "qlever-sweepexec-test")
            .string();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    const std::string basename = dir + "/idx";
    VectorIndexConfig cfg;
    cfg.name_ = "sweep";
    cfg.dimensions_ = kDim;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.scalar_ = VectorScalar::I8;
    cfg.rerankScalar_ = VectorScalar::Bf16;
    cfg.buildHnsw_ = false;
    VectorIndexBuilder builder{basename, cfg};
    std::mt19937 rng(4711);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> row(kDim);
    for (size_t i = 0; i < kN; ++i) {
      for (auto& x : row) {
        x = g(rng);
      }
      builder.add(mkId(100 + i), "<http://ex/v" + std::to_string(i) + ">",
                  row);
    }
    builder.build();
    index.open(basename, "sweep");
    query.resize(kDim);
    std::mt19937 qrng(31337);
    for (auto& x : query) {
      x = g(qrng);
    }
  }
};

ExecutorFixture& fixture() {
  static ExecutorFixture f;
  return f;
}

// ___________________________________________________________________________
// A plain whole-index two-layer i8 top-k gives IDENTICAL results with the
// executor on vs off -- for the blocked-sweep + heap route (k < 512), the
// blocked-sweep + histogram-select route (k >= 512, which also exercises the
// executor's float scratch), and the fine bf16 whole-index sweep.
TEST(VectorSweepExecutor, wholeIndexTopKIdenticalOnAndOff) {
  auto& f = fixture();
  for (size_t k : {size_t{100}, size_t{600}}) {
    std::vector<ScoredEntity> on, off;
    {
      SweepMasterEnv env{"on"};
      on = f.index.searchExactCoarse(f.query, k);
    }
    {
      SweepMasterEnv env{"off"};
      off = f.index.searchExactCoarse(f.query, k);
    }
    ASSERT_EQ(on.size(), off.size()) << "k=" << k;
    ASSERT_EQ(on.size(), k);
    for (size_t i = 0; i < k; ++i) {
      EXPECT_EQ(on[i].entity_, off[i].entity_) << "k=" << k << " i=" << i;
      EXPECT_EQ(on[i].distance_, off[i].distance_) << "k=" << k << " i=" << i;
    }
  }
  // The fine (bf16) whole-index sweep.
  std::vector<ScoredEntity> fineOn, fineOff;
  {
    SweepMasterEnv env{"on"};
    fineOn = f.index.searchExact(f.query, 50);
  }
  {
    SweepMasterEnv env{"off"};
    fineOff = f.index.searchExact(f.query, 50);
  }
  ASSERT_EQ(fineOn.size(), fineOff.size());
  for (size_t i = 0; i < fineOn.size(); ++i) {
    EXPECT_EQ(fineOn[i].entity_, fineOff[i].entity_) << i;
    EXPECT_EQ(fineOn[i].distance_, fineOff[i].distance_) << i;
  }
}

// ___________________________________________________________________________
// A task's exception is rethrown at the caller.
TEST(VectorSweepExecutor, taskExceptionPropagatesToCaller) {
  SweepMasterEnv env{"on"};
  EXPECT_THROW(
      runVectorSweep({}, [] { throw std::runtime_error("boom"); }),
      std::runtime_error);
  // The executor survives the exception and keeps running tasks.
  bool ran = false;
  runVectorSweep({}, [&] { ran = true; });
  EXPECT_TRUE(ran);
}

// ___________________________________________________________________________
// A task running ON the executor runs nested sweeps INLINE on the same
// thread -- with the default pool size 1 a re-queue would self-deadlock, so
// finishing at all (with the right thread identities) is the proof.
TEST(VectorSweepExecutor, nestedSweepRunsInlineOnExecutorThread) {
  SweepMasterEnv env{"on"};
  EXPECT_FALSE(SweepExecutor::onExecutorThread());
  bool outerOnExecutor = false;
  bool innerRan = false;
  bool innerOnSameThread = false;
  runVectorSweep({}, [&] {
    outerOnExecutor = SweepExecutor::onExecutorThread();
    const std::thread::id outerId = std::this_thread::get_id();
    runVectorSweep({}, [&] {
      innerRan = true;
      innerOnSameThread = std::this_thread::get_id() == outerId &&
                          SweepExecutor::onExecutorThread();
    });
  });
  EXPECT_TRUE(outerOnExecutor);
  EXPECT_TRUE(innerRan);
  EXPECT_TRUE(innerOnSameThread);
}

// ___________________________________________________________________________
// Per-worker scratch: non-null (and reused) on the executor, null off it.
TEST(VectorSweepExecutor, scratchOnlyOnExecutorThread) {
  SweepMasterEnv env{"on"};
  EXPECT_EQ(SweepExecutor::floatScratch(16), nullptr);
  EXPECT_EQ(SweepExecutor::int32Scratch(16), nullptr);
  float* first = nullptr;
  runVectorSweep({}, [&] {
    first = SweepExecutor::floatScratch(1024);
    ASSERT_NE(first, nullptr);
    int32_t* ints = SweepExecutor::int32Scratch(1024);
    ASSERT_NE(ints, nullptr);
    // Distinct arenas: writing one must not touch the other.
    first[0] = 1.5f;
    ints[0] = 42;
    EXPECT_EQ(first[0], 1.5f);
    EXPECT_EQ(ints[0], 42);
  });
  if (SweepExecutor::instance().numWorkers() == 1) {
    // One worker -> the next task gets the SAME (grown-once) buffer.
    runVectorSweep({}, [&] {
      EXPECT_EQ(SweepExecutor::floatScratch(512), first);
    });
  }
}

// ___________________________________________________________________________
// A caller whose task is still QUEUED behind a running sweep abandons the
// wait promptly when its cancellation callback throws -- and the abandoned
// task NEVER runs.
TEST(VectorSweepExecutor, cancelledWaiterAbandonsQueuedTask) {
  SweepMasterEnv env{"on"};
  auto& ex = SweepExecutor::instance();
  if (ex.numWorkers() != 1) {
    GTEST_SKIP() << "pool size != 1 (QLEVER_VECTOR_SWEEP_EXECUTORS set); the "
                    "queued shape needs exactly one worker";
  }
  // Occupy the single worker with a blocker task (submitted from a helper
  // thread, which blocks in `run` until the blocker is released).
  std::mutex m;
  std::condition_variable cv;
  bool blockerStarted = false;
  bool release = false;
  std::thread holder{[&] {
    runVectorSweep({}, [&] {
      {
        std::lock_guard<std::mutex> lock{m};
        blockerStarted = true;
      }
      cv.notify_all();
      std::unique_lock<std::mutex> lock{m};
      cv.wait(lock, [&] { return release; });
    });
  }};
  {
    std::unique_lock<std::mutex> lock{m};
    cv.wait(lock, [&] { return blockerStarted; });
  }
  // Submit a second task; its waiter is cancelled while the task is queued.
  std::atomic<bool> secondRan{false};
  const auto start = std::chrono::steady_clock::now();
  EXPECT_THROW(
      runVectorSweep([] { throw std::runtime_error("cancelled"); },
                     [&] { secondRan.store(true); }),
      std::runtime_error);
  const auto waited = std::chrono::steady_clock::now() - start;
  // "Promptly": a few cancellation-poll periods, not the blocker's lifetime.
  EXPECT_LT(std::chrono::duration_cast<std::chrono::milliseconds>(waited)
                .count(),
            1000);
  // Release the blocker; the worker then skips the abandoned entry.
  {
    std::lock_guard<std::mutex> lock{m};
    release = true;
  }
  cv.notify_all();
  holder.join();
  // Prove the abandoned task is skipped: run another task through the SAME
  // worker; by FIFO order it would run after the abandoned one.
  bool thirdRan = false;
  runVectorSweep({}, [&] { thirdRan = true; });
  EXPECT_TRUE(thirdRan);
  EXPECT_FALSE(secondRan.load());
}

}  // namespace
