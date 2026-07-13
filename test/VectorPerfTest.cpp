// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Performance benchmark of the exact whole-index vector scan (the brute-force
// top-k sweep of `VectorIndex::searchExact*`). DISABLED by default -- it
// builds a multi-GB synthetic index and runs for minutes. Run explicitly:
//
//   QLeverAllUnitTestsMain --gtest_also_run_disabled_tests
//       --gtest_filter='VectorPerf.*'   (one command line)
//
// Environment knobs:
//   VECTOR_PERF_N     number of vectors        (default 2140516, the
//                     production image index this benchmark mirrors)
//   VECTOR_PERF_DIM   dimensions               (default 1152, SigLIP2-so400m)
//   VECTOR_PERF_DIR   directory for the cached on-disk index (default
//                     $TMPDIR/qlever-vecperf-cache). The index is REUSED
//                     across runs when the files already exist; delete the
//                     directory to force a rebuild.
//   VECTOR_PERF_REPS  timed repetitions per configuration (default 7)
//
// The index matches the production config: binary (1-bit sign / Hamming)
// coarse scan layer + bf16 cosine fine rerank layer, no HNSW.

#ifdef _OPENMP
#include <omp.h>
#endif

#include <gtest/gtest.h>
#include <stdlib.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"

using namespace qlever::vector;

namespace {

// ___________________________________________________________________________
size_t envSize(const char* name, size_t dflt) {
  const char* v = getenv(name);
  if (v == nullptr || *v == '\0') {
    return dflt;
  }
  return static_cast<size_t>(strtoull(v, nullptr, 10));
}

std::string envStr(const char* name, std::string dflt) {
  const char* v = getenv(name);
  return (v == nullptr || *v == '\0') ? dflt : std::string{v};
}

// splitmix64: fast deterministic random bits for the synthetic vectors.
struct SplitMix64 {
  uint64_t state_;
  explicit SplitMix64(uint64_t seed) : state_{seed} {}
  uint64_t next() {
    uint64_t z = (state_ += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  }
  // Two floats in [-1, 1) per 64-bit draw.
  void fill(ql::span<float> out) {
    size_t i = 0;
    for (; i + 2 <= out.size(); i += 2) {
      uint64_t bits = next();
      out[i] = static_cast<int32_t>(bits) * (1.0f / 2147483648.0f);
      out[i + 1] =
          static_cast<int32_t>(bits >> 32) * (1.0f / 2147483648.0f);
    }
    if (i < out.size()) {
      out[i] = static_cast<int32_t>(next()) * (1.0f / 2147483648.0f);
    }
  }
};

Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

double toMs(std::chrono::steady_clock::duration d) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(d).count() /
         1e6;
}

// Time one call of `fn`, in milliseconds.
template <typename Fn>
double timeMs(Fn&& fn) {
  auto start = std::chrono::steady_clock::now();
  fn();
  return toMs(std::chrono::steady_clock::now() - start);
}

// Run `fn` `reps` times, return {min, median} milliseconds.
template <typename Fn>
std::pair<double, double> timeReps(size_t reps, Fn&& fn) {
  std::vector<double> ms;
  ms.reserve(reps);
  for (size_t r = 0; r < reps; ++r) {
    ms.push_back(timeMs(fn));
  }
  std::sort(ms.begin(), ms.end());
  return {ms.front(), ms[ms.size() / 2]};
}

struct BenchSetup {
  size_t n;
  size_t dim;
  size_t reps;
  std::string dir;
  std::string basename;  // `dir`/idx  (index name "perf")
  std::vector<float> query;

  size_t coarseRowBytes() const { return dim / 8; }
  size_t fineRowBytes() const { return dim * 2; }
};

// Build (or reuse a cached) synthetic two-layer index: binary scan layer +
// bf16 rerank layer, cosine, no HNSW, no csls.
BenchSetup setupIndex() {
  BenchSetup s;
  s.n = envSize("VECTOR_PERF_N", 2140516);
  s.dim = envSize("VECTOR_PERF_DIM", 1152);
  s.reps = envSize("VECTOR_PERF_REPS", 7);
  s.dir = envStr(
      "VECTOR_PERF_DIR",
      (std::filesystem::temp_directory_path() / "qlever-vecperf-cache")
          .string());
  std::filesystem::create_directories(s.dir);
  s.basename = s.dir + "/idx";

  // A deterministic random query (seed disjoint from the row seeds).
  s.query.resize(s.dim);
  SplitMix64 queryRng{0xabcdef1234567890ULL};
  queryRng.fill(s.query);

  const std::string metaFile = vectorMetaFile(s.basename, "perf");
  if (std::filesystem::exists(metaFile)) {
    printf("[setup] reusing cached index at %s (delete to rebuild)\n",
           s.dir.c_str());
    return s;
  }

  printf("[setup] building synthetic index: n=%zu dim=%zu (binary + bf16)\n",
         s.n, s.dim);
  fflush(stdout);
  VectorIndexConfig cfg;
  cfg.name_ = "perf";
  cfg.dimensions_ = static_cast<uint32_t>(s.dim);
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  cfg.buildHnsw_ = false;
  auto buildStart = std::chrono::steady_clock::now();
  VectorIndexBuilder builder{s.basename, cfg};
  std::vector<float> row(s.dim);
  std::string iri;
  for (size_t i = 0; i < s.n; ++i) {
    SplitMix64 rng{0x1234567800000000ULL + i};
    rng.fill(row);
    iri = "<http://ex/v" + std::to_string(i) + ">";
    builder.add(mkId(100 + i), iri, row);
    if ((i + 1) % 250000 == 0) {
      printf("[setup]   added %zu rows (%.1f s)\n", i + 1,
             toMs(std::chrono::steady_clock::now() - buildStart) / 1000.0);
      fflush(stdout);
    }
  }
  printf("[setup]   gathering (build())...\n");
  fflush(stdout);
  builder.build();
  printf("[setup] build done in %.1f s\n",
         toMs(std::chrono::steady_clock::now() - buildStart) / 1000.0);
  fflush(stdout);
  return s;
}

// Force the OpenMP max-thread ICV (the scan reads it via
// `vectorSearchThreadCap`). No-op without OpenMP.
void setThreads([[maybe_unused]] int t) {
#ifdef _OPENMP
  omp_set_num_threads(t);
#endif
}

int maxHwThreads() {
#ifdef _OPENMP
  return static_cast<int>(physicalCoreCount());
#else
  return 1;
#endif
}

// Pretty-print one measured configuration.
void report(const char* label, size_t n, size_t rowBytes, double minMs,
            double medianMs) {
  const double nsPerVec = minMs * 1e6 / static_cast<double>(n);
  const double gbps = static_cast<double>(n) * static_cast<double>(rowBytes) /
                      (minMs * 1e6);  // bytes/ns == GB/s
  printf("%-44s min %8.2f ms  median %8.2f ms  %6.2f ns/vec  %7.2f GB/s\n",
         label, minMs, medianMs, nsPerVec, gbps);
  fflush(stdout);
}

}  // namespace

// ___________________________________________________________________________
// The main benchmark: cold vs warm, thread-scaling sweep, residency modes,
// coarse (binary Hamming) and fine (bf16 cosine) layers, and the
// production-shaped covering-candidate-set call. Also asserts the
// serial/parallel result equivalence on the way.
TEST(VectorPerf, DISABLED_wholeIndexScanBenchmark) {
  // Let the benchmark drive the thread count purely via
  // `omp_set_num_threads` (the env override is memoized on first use, so it
  // must be set before the first search).
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);

  BenchSetup s = setupIndex();
  const size_t k = 500;  // production coarse pass: rerankK for binary, k=10
  const int maxT = maxHwThreads();

  printf("\n=== VectorPerf: n=%zu dim=%zu coarseRow=%zuB fineRow=%zuB "
         "maxThreads=%d reps=%zu ===\n",
         s.n, s.dim, s.coarseRowBytes(), s.fineRowBytes(), maxT, s.reps);

  // --------------------------------------------------------------------
  // 1. COLD scan: a fresh mmap (empty page tables; data may still be in the
  //    OS page cache, so this measures the fault/-TLB cost of a server
  //    restart on a warm host, not disk I/O).
  {
    VectorIndex idx;
    idx.open(s.basename, "perf");
    setThreads(maxT);
    double coldMs = timeMs([&] { idx.searchExactCoarse(s.query, k); });
    report("coarse sweep COLD (fresh mmap, maxT)", s.n, s.coarseRowBytes(),
           coldMs, coldMs);
    double warm1 = timeMs([&] { idx.searchExactCoarse(s.query, k); });
    report("coarse sweep 2nd scan (same mmap)", s.n, s.coarseRowBytes(),
           warm1, warm1);
  }

  // --------------------------------------------------------------------
  // 2. The persistent instance for all warm measurements.
  VectorIndex idx;
  idx.open(s.basename, "perf");
  // Fault everything in once so the "warm" numbers below are pure compute.
  setThreads(maxT);
  (void)idx.searchExactCoarse(s.query, k);
  (void)idx.searchExact(s.query, k);

  // --------------------------------------------------------------------
  // 3. Thread-scaling sweep on the warm COARSE layer (mmap residency).
  printf("\n--- coarse layer (binary/Hamming, %zu B/row), warm, mmap ---\n",
         s.coarseRowBytes());
  double coarse1T = 0;
  for (int t = 1; t <= maxT; t *= 2) {
    setThreads(t);
    auto [mn, md] = timeReps(s.reps, [&] { idx.searchExactCoarse(s.query, k); });
    char label[64];
    snprintf(label, sizeof label, "coarse sweep warm, %2d thread(s)", t);
    report(label, s.n, s.coarseRowBytes(), mn, md);
    if (t == 1) coarse1T = mn;
    if (t != maxT && t * 2 > maxT) {
      t = maxT / 2;  // make the last iteration exactly maxT
    }
  }
  setThreads(maxT);
  auto [coarseMaxT, coarseMaxTMed] =
      timeReps(s.reps, [&] { idx.searchExactCoarse(s.query, k); });
  printf("    speedup %d threads vs 1: %.1fx\n", maxT,
         coarse1T / coarseMaxT);

  // --------------------------------------------------------------------
  // 4. Fine layer (bf16 cosine) warm sweep.
  printf("\n--- fine layer (bf16 cosine, %zu B/row), warm, mmap ---\n",
         s.fineRowBytes());
  for (int t : {1, maxT}) {
    setThreads(t);
    auto [mn, md] = timeReps(s.reps, [&] { idx.searchExact(s.query, k); });
    char label[64];
    snprintf(label, sizeof label, "fine sweep warm, %2d thread(s)", t);
    report(label, s.n, s.fineRowBytes(), mn, md);
  }

  // --------------------------------------------------------------------
  // 5. Production shape: `searchExactCoarse` with a candidate set covering
  //    every live vector (bound-set join) -- includes the merge-join
  //    marshalling that runs before the parallel sweep.
  printf("\n--- covering candidate set (production shape) ---\n");
  std::vector<Id> allIds(idx.numLiveVectors());
  idx.memberEntities(allIds);
  setThreads(maxT);
  {
    auto [mn, md] = timeReps(s.reps, [&] {
      idx.searchExactCoarse(s.query, k, ql::span<const Id>{allIds});
    });
    report("coarse covering-candidates, maxT", s.n, s.coarseRowBytes(), mn, md);
    printf("    marshalling overhead vs bare sweep: %+.2f ms\n",
           mn - coarseMaxT);
  }

  // --------------------------------------------------------------------
  // 6. Residency::AlignedCopy (64B-aligned RAM copy, padded stride).
  printf("\n--- Residency::AlignedCopy (RAM copy) ---\n");
  {
    VectorIndex idxA;
    idxA.open(s.basename, "perf", VectorIndex::Residency::AlignedCopy,
              VectorIndex::Residency::AlignedCopy);
    setThreads(maxT);
    (void)idxA.searchExactCoarse(s.query, k);
    (void)idxA.searchExact(s.query, k);
    auto [mnC, mdC] =
        timeReps(s.reps, [&] { idxA.searchExactCoarse(s.query, k); });
    report("coarse sweep warm, maxT, AlignedCopy", s.n, s.coarseRowBytes(),
           mnC, mdC);
    auto [mnF, mdF] = timeReps(s.reps, [&] { idxA.searchExact(s.query, k); });
    report("fine sweep warm, maxT, AlignedCopy", s.n, s.fineRowBytes(), mnF,
           mdF);

    // Correctness: the aligned copy must return the identical top-k.
    setThreads(maxT);
    auto viaMmap = idx.searchExact(s.query, 10);
    auto viaCopy = idxA.searchExact(s.query, 10);
    ASSERT_EQ(viaMmap.size(), viaCopy.size());
    for (size_t i = 0; i < viaMmap.size(); ++i) {
      EXPECT_EQ(viaMmap[i].entity_, viaCopy[i].entity_) << i;
      EXPECT_EQ(viaMmap[i].distance_, viaCopy[i].distance_) << i;
    }
  }

  // --------------------------------------------------------------------
  // 7. Correctness: serial vs parallel top-k.
  //    Fine layer (distinct float distances): bit-identical result required.
  {
    setThreads(1);
    auto serial = idx.searchExact(s.query, 10);
    setThreads(maxT);
    auto parallel = idx.searchExact(s.query, 10);
    ASSERT_EQ(serial.size(), parallel.size());
    for (size_t i = 0; i < serial.size(); ++i) {
      EXPECT_EQ(serial[i].entity_, parallel[i].entity_) << i;
      EXPECT_EQ(serial[i].distance_, parallel[i].distance_) << i;
    }
    // Coarse layer: integer Hamming distances tie heavily, so the ENTITY
    // choice at the k-boundary legitimately depends on the thread count;
    // the sorted DISTANCE multiset must match exactly.
    setThreads(1);
    auto serialC = idx.searchExactCoarse(s.query, k);
    setThreads(maxT);
    auto parallelC = idx.searchExactCoarse(s.query, k);
    ASSERT_EQ(serialC.size(), parallelC.size());
    for (size_t i = 0; i < serialC.size(); ++i) {
      EXPECT_EQ(serialC[i].distance_, parallelC[i].distance_) << i;
    }

    // NEAR-covering candidate set (all live vectors but one that is not in
    // the top-k): must take the scattered-gather fallback, not the
    // whole-index sweep, and still return the identical fine top-k.
    setThreads(maxT);
    auto whole = idx.searchExact(s.query, 10);
    size_t excluded = 0;
    auto inTop = [&](Id id) {
      return std::any_of(whole.begin(), whole.end(),
                         [&](const ScoredEntity& e) { return e.entity_ == id; });
    };
    while (excluded < allIds.size() && inTop(allIds[excluded])) {
      ++excluded;
    }
    ASSERT_LT(excluded, allIds.size());
    std::vector<Id> nearCover;
    nearCover.reserve(allIds.size() - 1);
    for (size_t i = 0; i < allIds.size(); ++i) {
      if (i != excluded) {
        nearCover.push_back(allIds[i]);
      }
    }
    auto viaGather =
        idx.searchExact(s.query, 10, ql::span<const Id>{nearCover});
    ASSERT_EQ(whole.size(), viaGather.size());
    for (size_t i = 0; i < whole.size(); ++i) {
      EXPECT_EQ(whole[i].entity_, viaGather[i].entity_) << i;
      EXPECT_EQ(whole[i].distance_, viaGather[i].distance_) << i;
    }
  }
  printf("\n[done] correctness checks passed\n");
}

// ___________________________________________________________________________
// A steady-state loop for external profilers (`perf stat` / `perf record`):
// repeats one configuration until VECTOR_PERF_SECONDS wall seconds elapsed.
//   VECTOR_PERF_LAYER    coarse (default) | fine
//   VECTOR_PERF_THREADS  thread count (default: all physical cores)
//   VECTOR_PERF_SECONDS  duration (default 10)
//   VECTOR_PERF_RESIDENCY none (default) | aligned
TEST(VectorPerf, DISABLED_profileLoop) {
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);
  BenchSetup s = setupIndex();
  const size_t k = envSize("VECTOR_PERF_K", 500);
  const bool fine = envStr("VECTOR_PERF_LAYER", "coarse") == "fine";
  const bool aligned = envStr("VECTOR_PERF_RESIDENCY", "none") == "aligned";
  const int threads =
      static_cast<int>(envSize("VECTOR_PERF_THREADS",
                               static_cast<size_t>(maxHwThreads())));
  const double seconds =
      static_cast<double>(envSize("VECTOR_PERF_SECONDS", 10));

  VectorIndex idx;
  auto res = aligned ? VectorIndex::Residency::AlignedCopy
                     : VectorIndex::Residency::None;
  idx.open(s.basename, "perf", res, res);
  setThreads(threads);
  // Warm up once.
  (void)(fine ? idx.searchExact(s.query, k)
              : idx.searchExactCoarse(s.query, k));
  printf("[profileLoop] layer=%s threads=%d residency=%s for %.0f s\n",
         fine ? "fine" : "coarse", threads, aligned ? "aligned" : "none",
         seconds);
  fflush(stdout);
  size_t iters = 0;
  auto start = std::chrono::steady_clock::now();
  double elapsed = 0;
  while (elapsed < seconds) {
    (void)(fine ? idx.searchExact(s.query, k)
                : idx.searchExactCoarse(s.query, k));
    ++iters;
    elapsed = toMs(std::chrono::steady_clock::now() - start) / 1000.0;
  }
  const double msPerScan = elapsed * 1000.0 / static_cast<double>(iters);
  const size_t rowBytes = fine ? s.fineRowBytes() : s.coarseRowBytes();
  report("profileLoop steady state", s.n, rowBytes, msPerScan, msPerScan);
  printf("[profileLoop] %zu scans in %.1f s\n", iters, elapsed);
}

// ___________________________________________________________________________
// A/B benchmark of the `vec:bf16Kernel` selector: full-precision WHOLE-INDEX
// fine sweep (`searchExact`) and the two-layer SCATTERED rerank of the
// coarse-best rows (`searchExactByRows`), each measured for the SIMD, the
// fixed-AMX, and the current (Auto = NumKong GEMM) kernels.
//   VECTOR_PERF_THREADS  thread count (default: all physical cores)
//   VECTOR_PERF_K        top-k (default 1000)
//   VECTOR_PERF_RERANKK  rerank batch size for the scattered pass (default 4096)
//   VECTOR_PERF_RESIDENCY none (default) | aligned
TEST(VectorPerf, DISABLED_bf16KernelAbBenchmark) {
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);
  BenchSetup s = setupIndex();
  const size_t k = envSize("VECTOR_PERF_K", 1000);
  const size_t rerankK = envSize("VECTOR_PERF_RERANKK", 4096);
  const bool aligned = envStr("VECTOR_PERF_RESIDENCY", "none") == "aligned";
  const int threads = static_cast<int>(
      envSize("VECTOR_PERF_THREADS", static_cast<size_t>(maxHwThreads())));
  auto residency = aligned ? VectorIndex::Residency::AlignedCopy
                           : VectorIndex::Residency::None;
  VectorIndex idx;
  idx.open(s.basename, "perf", residency, residency);
  setThreads(threads);

  using qlever::vector::Bf16Kernel;
  struct KernSpec {
    const char* name;
    Bf16Kernel kern;
  };
  // `Auto` now resolves to the fixed-AMX block on this CPU (the new default),
  // so it and `Amx` coincide here; `Punned` is the pre-change per-row baseline.
  // (The NumKong GEMM -- the OLD `Auto` -- is no longer selectable by a flag;
  // compare against a git-HEAD binary for that number.)
  const KernSpec specs[] = {{"auto (=amx here)", Bf16Kernel::Auto},
                            {"amx (fixed)", Bf16Kernel::Amx},
                            {"simd", Bf16Kernel::Simd},
                            {"punned (per-row)", Bf16Kernel::Punned}};

  printf("\n=== vec:bf16Kernel A/B: n=%zu dim=%zu fineRow=%zuB threads=%d "
         "k=%zu rerankK=%zu residency=%s ===\n",
         s.n, s.dim, s.fineRowBytes(), threads, k, rerankK,
         aligned ? "aligned" : "none");

  // Cross-kernel result agreement (top-10 distances) before timing.
  auto ref = idx.searchExact(s.query, 10, std::nullopt, std::nullopt, {},
                             nullptr, Bf16Kernel::Punned);
  for (const KernSpec& sp : specs) {
    auto got = idx.searchExact(s.query, 10, std::nullopt, std::nullopt, {},
                               nullptr, sp.kern);
    ASSERT_EQ(got.size(), ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
      EXPECT_NEAR(got[i].distance_, ref[i].distance_, 1e-6f)
          << sp.name << " i=" << i;
    }
  }
  printf("[agreement] all kernels match the punned top-10 to 1e-6\n");

  // (1) FULL-PRECISION whole-index fine sweep.
  printf("\n--- full-precision whole-index fine sweep (searchExact) ---\n");
  for (const KernSpec& sp : specs) {
    // Warm up.
    (void)idx.searchExact(s.query, k, std::nullopt, std::nullopt, {}, nullptr,
                          sp.kern);
    auto [mn, md] = timeReps(s.reps, [&] {
      (void)idx.searchExact(s.query, k, std::nullopt, std::nullopt, {}, nullptr,
                            sp.kern);
    });
    char label[80];
    snprintf(label, sizeof label, "full-scan  %-20s", sp.name);
    report(label, s.n, s.fineRowBytes(), mn, md);
  }

  // (2) SCATTERED rerank: coarse-rank rerankK rows, then rerank them on the
  //     fine bf16 layer via each kernel (the multi-row SIMD gather).
  printf("\n--- scattered rerank of %zu coarse-best rows (searchExactByRows) "
         "---\n",
         rerankK);
  auto coarseRows = idx.searchExactCoarseWithRows(s.query, rerankK);
  ql::span<const qlever::vector::ScoredRow> rowsSpan{coarseRows};
  for (const KernSpec& sp : specs) {
    (void)idx.searchExactByRows(s.query, k, rowsSpan, std::nullopt, {},
                                sp.kern);
    auto [mn, md] = timeReps(s.reps, [&] {
      (void)idx.searchExactByRows(s.query, k, rowsSpan, std::nullopt, {},
                                  sp.kern);
    });
    char label[80];
    snprintf(label, sizeof label, "rerank     %-20s", sp.name);
    // Report GB/s over the reranked bytes (rerankK rows, not the whole index).
    report(label, coarseRows.size(), s.fineRowBytes(), mn, md);
  }
}

// ___________________________________________________________________________
// Round-2 benchmark: the CSLS coarse sweep, the softmax cut, the large-rerank
// gather, and -- the key puzzle -- the covering fast path with a candidate
// span that is a large SUPERSET of the live set (extras = non-members), which
// the Round-1 exact-cover benchmark did not exercise.
TEST(VectorPerf, DISABLED_scanPathsBenchmark) {
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);
  BenchSetup s = setupIndex();
  const int maxT = maxHwThreads();
  // The production box caps at 8 threads; measure at that too.
  const int prodT = std::min(8, maxT);

  VectorIndex idx;
  idx.open(s.basename, "perf");
  setThreads(maxT);
  (void)idx.searchExactCoarse(s.query, 500);  // warm everything in

  // The live members (ascending Id == mkId(100..100+n-1)).
  std::vector<Id> allIds(idx.numLiveVectors());
  idx.memberEntities(allIds);
  ASSERT_EQ(allIds.size(), s.n);

  // A SUPERSET covering candidate set: every live id + ~1M non-member ids
  // (mkId(100+n ..), which are valid VocabIndex ids absent from the index).
  // Already ascending (members are 100..100+n-1, extras follow), so the fast
  // path's coverage check sees a sorted span.
  const size_t extras = s.n / 2;  // 2.14M live -> ~3.21M candidates
  std::vector<Id> superset;
  superset.reserve(s.n + extras);
  superset.insert(superset.end(), allIds.begin(), allIds.end());
  for (size_t i = 0; i < extras; ++i) {
    superset.push_back(mkId(100 + s.n + i));
  }

  printf("\n=== VectorPerf scan paths: n=%zu extras=%zu maxT=%d prodT=%d ===\n",
         s.n, extras, maxT, prodT);

  for (int t : {prodT, maxT}) {
    setThreads(t);
    printf("\n--- %d threads ---\n", t);
    // (a) plain whole-index coarse (no candidates) -- the pure sweep.
    auto [swMn, swMd] =
        timeReps(s.reps, [&] { idx.searchExactCoarse(s.query, 500); });
    report("plain coarse sweep (no candidates)", s.n, s.coarseRowBytes(), swMn,
           swMd);
    // (a2) plain coarse sweep selecting a LARGE k (a rerank-floor sized top-M):
    // exposes the O(threads x k) heap-merge cost of a big selection.
    auto [sw2Mn, sw2Md] =
        timeReps(s.reps, [&] { idx.searchExactCoarse(s.query, 10000); });
    report("plain coarse sweep k=10000", s.n, s.coarseRowBytes(), sw2Mn, sw2Md);
    // (b) exact 2.14M covering candidates -> coverage(n) + sweep.
    auto [exMn, exMd] = timeReps(s.reps, [&] {
      idx.searchExactCoarse(s.query, 500, ql::span<const Id>{allIds});
    });
    report("coarse, exact-cover candidates (=n)", s.n, s.coarseRowBytes(), exMn,
           exMd);
    printf("    coverage-check(%zu) cost ~= %+.2f ms\n", allIds.size(),
           exMn - swMn);
    // (c) 3.1M SUPERSET covering candidates -> coverage(m>n) + sweep.
    auto [suMn, suMd] = timeReps(s.reps, [&] {
      idx.searchExactCoarse(s.query, 500, ql::span<const Id>{superset});
    });
    report("coarse, SUPERSET candidates (1.5n)", s.n, s.coarseRowBytes(), suMn,
           suMd);
    printf("    coverage-check(%zu) cost ~= %+.2f ms\n", superset.size(),
           suMn - swMn);
  }

  // -------------------------------------------------------------------
  // CSLS softmax cut (no sidecar needed): the whole hot pipeline of query B
  // -- full coarse sweep of every live row + a bounded fine rerank. softmaxN
  // ~ the production rerank width.
  printf("\n--- CSLS softmax cut (coarse sweep + bounded rerank) ---\n");
  CslsCut softmax;
  softmax.mode_ = CslsCut::Mode::Softmax;
  softmax.softmaxN_ = 1000;
  for (int t : {prodT, maxT}) {
    setThreads(t);
    auto [mn, md] = timeReps(s.reps, [&] {
      size_t scored = 0;
      idx.searchCsls(s.query, softmax, 10, std::nullopt, std::nullopt, {},
                     &scored);
    });
    char label[64];
    snprintf(label, sizeof label, "CSLS softmax whole-index, %2d thr", t);
    report(label, s.n, s.coarseRowBytes(), mn, md);
    auto [cmn, cmd] = timeReps(s.reps, [&] {
      idx.searchCsls(s.query, softmax, 10, ql::span<const Id>{superset},
                     std::nullopt, {});
    });
    snprintf(label, sizeof label, "CSLS softmax SUPERSET cand, %2d thr", t);
    report(label, s.n, s.coarseRowBytes(), cmn, cmd);
  }

  // -------------------------------------------------------------------
  // Large-rerank scattered gather over the FINE layer (query C's 50k rerank).
  printf("\n--- fine-layer scattered gather (50k candidates) ---\n");
  std::vector<Id> gather50k;
  gather50k.reserve(50000);
  for (size_t i = 0; i < 50000; ++i) {
    gather50k.push_back(allIds[(i * 41) % allIds.size()]);
  }
  std::sort(gather50k.begin(), gather50k.end());
  gather50k.erase(std::unique(gather50k.begin(), gather50k.end()),
                  gather50k.end());
  for (int t : {prodT, maxT}) {
    setThreads(t);
    auto [mn, md] = timeReps(s.reps, [&] {
      idx.searchExact(s.query, 1000, ql::span<const Id>{gather50k});
    });
    char label[64];
    snprintf(label, sizeof label, "fine gather %zu cand, %2d thr",
             gather50k.size(), t);
    report(label, gather50k.size(), s.fineRowBytes(), mn, md);
  }

  // -------------------------------------------------------------------
  // Correctness. (1) The SUPERSET covering fast path returns exactly the
  // plain whole-index top-k (no false negatives that would change results,
  // no false positives that would score a non-member).
  {
    setThreads(maxT);
    auto plain = idx.searchExactCoarse(s.query, 500);
    auto viaSuper =
        idx.searchExactCoarse(s.query, 500, ql::span<const Id>{superset});
    ASSERT_EQ(plain.size(), viaSuper.size());
    for (size_t i = 0; i < plain.size(); ++i) {
      EXPECT_EQ(plain[i].distance_, viaSuper[i].distance_) << i;
    }
  }
  // (2) CSLS softmax survivors are thread-count invariant (the coarse sweep
  // fills a per-index array, so 1 vs N threads must be bit-identical).
  {
    setThreads(1);
    auto serial = idx.searchCsls(s.query, softmax, 10);
    setThreads(maxT);
    auto parallel = idx.searchCsls(s.query, softmax, 10);
    ASSERT_EQ(serial.size(), parallel.size());
    for (size_t i = 0; i < serial.size(); ++i) {
      EXPECT_EQ(serial[i].entity_, parallel[i].entity_) << i;
      EXPECT_EQ(serial[i].distance_, parallel[i].distance_) << i;
    }
    // And covering-candidates CSLS == whole-index CSLS.
    setThreads(maxT);
    auto viaCand =
        idx.searchCsls(s.query, softmax, 10, ql::span<const Id>{superset});
    ASSERT_EQ(parallel.size(), viaCand.size());
    for (size_t i = 0; i < parallel.size(); ++i) {
      EXPECT_EQ(parallel[i].entity_, viaCand[i].entity_) << i;
      EXPECT_EQ(parallel[i].distance_, viaCand[i].distance_) << i;
    }
  }
  printf("\n[done] scan-paths correctness checks passed\n");
}

// ___________________________________________________________________________
// Round-3: the `cslsThreshold`-shaped whole-index query -- coarse sweep +
// select the top-`cslsRerankFloor`=10,000, then a bounded fine rerank -- on the
// PRELOADED (Residency::AlignedCopy, no page faults) 2.14M binary+bf16 index.
// The coarse SELECTION at k=10,000 is the target. Set VECTOR_CSLS_PHASE=1 for
// the per-phase split (setup / coarseSweepSelect / rerank+rq / cut+sort).
TEST(VectorPerf, DISABLED_cslsSelectBenchmark) {
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);
  BenchSetup s = setupIndex();
  const int maxT = maxHwThreads();
  const int prodT = std::min(8, maxT);

  // Preload both layers into aligned RAM (matches the production `aligned`
  // residency: no mmap faults during the timed scans).
  VectorIndex idx;
  idx.open(s.basename, "perf", VectorIndex::Residency::AlignedCopy,
           VectorIndex::Residency::AlignedCopy);

  // Softmax cut needs no CSLS sidecar and shares the EXACT coarse
  // sweep+select(topM=cslsRerankFloor) of the threshold/knee cuts (only the
  // post-selection cut math differs, ~microseconds), so it faithfully measures
  // the `cslsThreshold` coarse cost. softmaxN small so widening never fires.
  CslsCut cut;
  cut.mode_ = CslsCut::Mode::Softmax;
  cut.softmaxN_ = 1000;

  setThreads(maxT);
  (void)idx.searchCsls(s.query, cut, 10);  // warm
  (void)idx.searchExactCoarse(s.query, 500);

  printf(
      "\n=== VectorPerf CSLS select: n=%zu floorM=10000 (aligned/preloaded) "
      "===\n",
      s.n);
  for (int t : {prodT, maxT}) {
    setThreads(t);
    // Pure coarse COMPUTE reference (running-pointer whole-index sweep, tiny
    // heap): the floor the CSLS coarse pass should approach.
    auto [pcMn, pcMd] =
        timeReps(s.reps, [&] { idx.searchExactCoarse(s.query, 500); });
    char label[80];
    snprintf(label, sizeof label, "  ref: plain coarse k=500, %2d thr", t);
    report(label, s.n, s.coarseRowBytes(), pcMn, pcMd);
    auto [mn, md] = timeReps(s.reps, [&] { idx.searchCsls(s.query, cut, 10); });
    snprintf(label, sizeof label, "CSLS whole-index (floorM 10000), %2d thr",
             t);
    report(label, s.n, s.coarseRowBytes(), mn, md);
  }

  // Correctness: histogram-select survivors are thread-count invariant and
  // match the covering-candidate path (both must be bit-identical to the
  // definitional (distance,index) coarse ranking).
  {
    setThreads(1);
    auto serial = idx.searchCsls(s.query, cut, 10);
    setThreads(maxT);
    auto parallel = idx.searchCsls(s.query, cut, 10);
    ASSERT_EQ(serial.size(), parallel.size());
    for (size_t i = 0; i < serial.size(); ++i) {
      EXPECT_EQ(serial[i].entity_, parallel[i].entity_) << i;
      EXPECT_EQ(serial[i].distance_, parallel[i].distance_) << i;
    }
    printf("[cslsSelect] survivors: %zu (1-thread == %d-thread)\n",
           serial.size(), maxT);
  }
  printf("[done] cslsSelect checks passed\n");
}

// ___________________________________________________________________________
// Round-4: the `vec:autoCut` mode-switch cost -- the cold reranked-to-plateau
// COMPUTE stage vs. re-applying a cut over the cached reranked set (what a
// mode switch does). On the preloaded 2.14M binary+bf16 index, cosine signal
// (needs no csls sidecar).
TEST(VectorPerf, DISABLED_autoCutModeSwitchBenchmark) {
  setenv("QLEVER_VECTOR_SEARCH_THREADS", "4096", 1);
  BenchSetup s = setupIndex();
  const int maxT = maxHwThreads();
  const int prodT = std::min(8, maxT);
  VectorIndex idx;
  idx.open(s.basename, "perf", VectorIndex::Residency::AlignedCopy,
           VectorIndex::Residency::AlignedCopy);
  const float ff = DEFAULT_ZCUT_FLOOR_FRACTION;
  const float wf = DEFAULT_ZCUT_FRACTION_BROAD + DEFAULT_ZCUT_WIDEN_MARGIN;
  // The production rerank cap (`resolveCslsCut` = cslsRerankFloor * 8). The
  // TOP-ANCHORED widen (fraction dial) reranks toward the floor for the broad
  // mode -- how deep is data-driven -- but never exceeds the cap.
  const size_t cap = idx.cslsRerankFloor() * 8;
  setThreads(maxT);
  (void)idx.computeCslsReranked(s.query, 10, ff, wf, cap);  // warm
  auto mkCut = [](float fraction) {
    CslsCut c;
    c.mode_ = CslsCut::Mode::ZCut;
    c.signal_ = CslsCut::Signal::Cosine;
    c.fraction_ = fraction;
    c.keepAtLeastOne_ = true;
    return c;
  };
  printf("\n=== autoCut mode-switch: n=%zu (aligned) ===\n", s.n);
  for (int t : {prodT, maxT}) {
    setThreads(t);
    // Cold: the whole reranked-to-plateau compute (a cache MISS).
    auto [cMn, cMd] = timeReps(s.reps, [&] {
      auto r = idx.computeCslsReranked(s.query, 10, ff, wf, cap);
      (void)r;
    });
    // Cached: re-apply a (different-mode) cut over an already-computed set.
    auto reranked = idx.computeCslsReranked(s.query, 10, ff, wf, cap);
    auto [aMn, aMd] = timeReps(s.reps, [&] {
      auto hits =
          idx.applyCslsCut(reranked, mkCut(DEFAULT_ZCUT_FRACTION_BROAD));
      (void)hits;
    });
    // The per-mode keep-count over the SAME cached reranked set: broad (f 0.85)
    // >> precise (f 0.3).
    const size_t keptPrecise =
        idx.applyCslsCut(reranked, mkCut(DEFAULT_ZCUT_FRACTION_PRECISE)).size();
    const size_t keptBroad =
        idx.applyCslsCut(reranked, mkCut(DEFAULT_ZCUT_FRACTION_BROAD)).size();
    char label[80];
    snprintf(label, sizeof label, "cold compute (miss), %2d thr", t);
    report(label, s.n, s.coarseRowBytes(), cMn, cMd);
    printf(
        "  mode-switch cut-only (cache hit), %2d thr: min %.3f ms  "
        "(reranked=%zu, keep precise=%zu broad=%zu, cap=%zu)\n",
        t, aMn, reranked.rerankDepth(), keptPrecise, keptBroad, cap);
  }
}
