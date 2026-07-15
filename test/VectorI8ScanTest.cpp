// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests of the i8 (int8 quantized-cosine) COARSE-layer scan fast path:
//   * the hand-rolled VNNI kernels (`VectorI8Kernels`): exact integer dots
//     (block, gather, pair -- including the -128 edge and non-multiple-of-64
//     dims) and the shared angular finalize;
//   * the O(n) float-histogram select (`selectSmallestPairsFloat`):
//     bit-identical to a full (distance, index) sort prefix, tie-heavy
//     inputs, thread-count independence;
//   * the two-layer i8+bf16 index end to end: the unified fast path vs the
//     punned-metric baseline (QLEVER_VECTOR_SEARCH_I8=off), the histogram vs
//     blocked-heap top-k routes, the `vec:i8Kernel` dial, the filtered
//     (scattered gather) path, and coarse+rerank recovering the exact top-k.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorI8Kernels.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorSelect.h"

namespace {
using namespace qlever::vector;

// ---------------------------------------------------------------------------
// Kernel-level tests (no index; skipped on a CPU without AVX-512-VNNI).

// Exact scalar reference for the signed i8 dot.
int32_t dotRef(const int8_t* a, const int8_t* b, size_t dim) {
  int32_t s = 0;
  for (size_t i = 0; i < dim; ++i) {
    s += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
  }
  return s;
}

// Deterministic i8 test data spanning the FULL range, including -128 and 127.
std::vector<int8_t> randomI8(size_t n, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> d(-128, 127);
  std::vector<int8_t> v(n);
  for (auto& x : v) {
    x = static_cast<int8_t>(d(rng));
  }
  // Plant the edges.
  if (n >= 2) {
    v[0] = -128;
    v[1] = 127;
  }
  return v;
}

TEST(VectorI8Kernels, rowSumsExactIncludingMinus128) {
  const auto v = randomI8(1000, 42);
  int32_t sum = 0, normSq = 0;
  i8kernels::rowSums(v.data(), v.size(), &sum, &normSq);
  int64_t refSum = 0, refSq = 0;
  for (int8_t x : v) {
    refSum += x;
    refSq += static_cast<int64_t>(x) * x;
  }
  EXPECT_EQ(sum, refSum);
  EXPECT_EQ(normSq, refSq);
}

TEST(VectorI8Kernels, dotPairMatchesScalarReference) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  for (size_t dim : {1u, 4u, 31u, 32u, 63u, 64u, 96u, 100u, 129u, 1152u}) {
    const auto a = randomI8(dim, 7 + dim);
    const auto b = randomI8(dim, 1000 + dim);
    EXPECT_EQ(i8kernels::dotPair(a.data(), b.data(), dim),
              dotRef(a.data(), b.data(), dim))
        << "dim=" << dim;
  }
}

// The block/gather kernels compute the UNSIGNED-BIASED dot; the identity
// `dotU - 128*sum(row) == dot` must hold exactly for every row, count-tail
// (`count % 8 != 0`) and dim-tail (`dim % 64 != 0`) combination.
TEST(VectorI8Kernels, blockAndGatherMatchExactIdentity) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  for (size_t dim : {4u, 64u, 100u, 129u}) {
    for (size_t count : {0u, 1u, 7u, 8u, 9u, 40u}) {
      const size_t stride = dim + 12;  // padded rows, like a strided store
      std::vector<int8_t> matrix(count * stride);
      std::vector<int32_t> sums(count);
      for (size_t r = 0; r < count; ++r) {
        auto row = randomI8(dim, static_cast<uint32_t>(500 + r + dim));
        std::copy(row.begin(), row.end(), matrix.begin() + r * stride);
        int32_t s = 0, s2 = 0;
        i8kernels::rowSums(row.data(), dim, &s, &s2);
        sums[r] = s;
      }
      const auto q = randomI8(dim, static_cast<uint32_t>(99 + dim));
      std::vector<uint8_t> qx(dim);
      for (size_t i = 0; i < dim; ++i) {
        qx[i] = static_cast<uint8_t>(static_cast<uint8_t>(q[i]) ^ 0x80u);
      }
      std::vector<int32_t> dotsU(count, 0);
      i8kernels::dotBlockVnni(qx.data(),
                              reinterpret_cast<const char*>(matrix.data()),
                              stride, count, dim, dotsU.data());
      for (size_t r = 0; r < count; ++r) {
        const int32_t dot = dotsU[r] - 128 * sums[r];
        EXPECT_EQ(dot, dotRef(q.data(), matrix.data() + r * stride, dim))
            << "block dim=" << dim << " count=" << count << " row=" << r;
      }
      // Gather over a reversed row order.
      std::vector<size_t> rowOf(count);
      for (size_t r = 0; r < count; ++r) {
        rowOf[r] = count - 1 - r;
      }
      std::vector<int32_t> dotsG(count, 0);
      i8kernels::dotGatherVnni(qx.data(),
                               reinterpret_cast<const char*>(matrix.data()),
                               stride, rowOf.data(), count, dim, dotsG.data());
      for (size_t r = 0; r < count; ++r) {
        const size_t src = rowOf[r];
        const int32_t dot = dotsG[r] - 128 * sums[src];
        EXPECT_EQ(dot, dotRef(q.data(), matrix.data() + src * stride, dim))
            << "gather dim=" << dim << " count=" << count << " row=" << r;
      }
    }
  }
}

TEST(VectorI8Kernels, angularFinalizeEdgeCasesAndAccuracy) {
  // Both zero vectors -> 0; zero dot -> 1 (NumKong's conventions).
  EXPECT_EQ(i8kernels::angularFinalize(0.f, 0.f, 0.f), 0.f);
  EXPECT_EQ(i8kernels::angularFinalize(0.f, 5.f, 7.f), 1.f);
  // A negative result is clamped at 0 (identical vectors, rsqrt rounding).
  EXPECT_GE(i8kernels::angularFinalize(1000.f, 999.9f, 999.9f), 0.f);
  // Accuracy vs the double-precision formula: the rsqrt + one Newton-Raphson
  // step is good to ~1e-7 relative; distances live in [0, 2].
  std::mt19937 rng(3);
  std::uniform_real_distribution<double> d(1.0, 4e6);
  std::uniform_real_distribution<double> c(-1.0, 1.0);
  for (int i = 0; i < 1000; ++i) {
    const double a2 = d(rng), b2 = d(rng);
    const double dot = c(rng) * std::sqrt(a2 * b2);
    const double ref = std::max(0.0, 1.0 - dot / std::sqrt(a2 * b2));
    const float got = i8kernels::angularFinalize(static_cast<float>(dot),
                                                 static_cast<float>(a2),
                                                 static_cast<float>(b2));
    EXPECT_NEAR(got, ref, 2e-6)
        << "a2=" << a2 << " b2=" << b2 << " dot=" << dot;
  }
}

// ---------------------------------------------------------------------------
// The O(n) float-histogram select: bit-identical to the full-sort prefix.

using Pair = std::pair<float, uint64_t>;

std::vector<Pair> selectByFullSort(const std::vector<float>& dists, size_t m) {
  std::vector<Pair> all(dists.size());
  for (size_t i = 0; i < dists.size(); ++i) {
    all[i] = {dists[i], static_cast<uint64_t>(i)};
  }
  std::sort(all.begin(), all.end());
  all.resize(std::min(m, all.size()));
  return all;
}

TEST(VectorSelect, selectSmallestPairsFloatMatchesFullSort) {
  std::mt19937 rng(11);
  // Heavy ties on purpose: distances quantized to a small grid, so both the
  // (distance, index) tiebreak and the boundary-bucket handling are load-
  // bearing, plus a uniform continuous batch.
  for (bool quantized : {true, false}) {
    std::uniform_real_distribution<float> d(0.f, 2.f);
    std::vector<float> dists(50'000);
    for (auto& x : dists) {
      x = d(rng);
      if (quantized) {
        x = std::round(x * 64.f) / 64.f;  // ~128 distinct values
      }
    }
    for (size_t m : {size_t{0}, size_t{1}, size_t{5}, size_t{100},
                     size_t{10'000}, dists.size(), dists.size() + 17}) {
      const auto ref = selectByFullSort(dists, m);
      for (int threads : {1, 4}) {
        const auto got =
            selectSmallestPairsFloat(dists.data(), dists.size(), m, threads);
        ASSERT_EQ(got.size(), ref.size())
            << "quantized=" << quantized << " m=" << m << " t=" << threads;
        for (size_t i = 0; i < got.size(); ++i) {
          ASSERT_EQ(got[i], ref[i]) << "quantized=" << quantized << " m=" << m
                                    << " t=" << threads << " i=" << i;
        }
      }
    }
  }
}

TEST(VectorSelect, selectSmallestPairsFloatEdgeCases) {
  // Empty input.
  EXPECT_TRUE(selectSmallestPairsFloat(nullptr, 0, 5, 1).empty());
  // All-equal distances: the m smallest indices, ascending.
  std::vector<float> same(1000, 0.75f);
  const auto got = selectSmallestPairsFloat(same.data(), same.size(), 7, 2);
  ASSERT_EQ(got.size(), 7u);
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i], (Pair{0.75f, i}));
  }
  // Values beyond the nominal [0, 2] range share the last bucket but stay
  // correctly ordered.
  std::vector<float> wide{2.5f, 0.1f, 2.4f, 0.2f};
  const auto w = selectSmallestPairsFloat(wide.data(), wide.size(), 3, 1);
  ASSERT_EQ(w.size(), 3u);
  EXPECT_EQ(w[0], (Pair{0.1f, 1}));
  EXPECT_EQ(w[1], (Pair{0.2f, 3}));
  EXPECT_EQ(w[2], (Pair{2.4f, 2}));
}

// ---------------------------------------------------------------------------
// End-to-end on a two-layer i8 (coarse) + bf16 (fine) index.

Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

// One shared synthetic index: 3000 rows (above the parallel threshold), dim
// 100 (NOT a multiple of 32/64 -> every kernel tail is exercised), fixed
// seeds. Built once; opened TWICE -- with the i8 fast path enabled (`on`) and
// disabled via QLEVER_VECTOR_SEARCH_I8=off (`off`, the punned-metric
// baseline).
struct I8Fixture {
  static constexpr size_t kN = 3000;
  static constexpr size_t kDim = 100;
  std::string basename;
  VectorIndex on;
  VectorIndex off;
  std::vector<float> query;

  I8Fixture() {
    const std::string dir =
        (std::filesystem::temp_directory_path() / "qlever-i8scan-test")
            .string();
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    basename = dir + "/idx";
    VectorIndexConfig cfg;
    cfg.name_ = "i8t";
    cfg.dimensions_ = kDim;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.scalar_ = VectorScalar::I8;
    cfg.rerankScalar_ = VectorScalar::Bf16;
    cfg.buildHnsw_ = false;
    VectorIndexBuilder builder{basename, cfg};
    std::mt19937 rng(2024);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> row(kDim);
    for (size_t i = 0; i < kN; ++i) {
      for (auto& x : row) {
        x = g(rng);
      }
      builder.add(mkId(100 + i), "<http://ex/v" + std::to_string(i) + ">", row);
    }
    builder.build();
    unsetenv("QLEVER_VECTOR_SEARCH_I8");
    on.open(basename, "i8t");
    setenv("QLEVER_VECTOR_SEARCH_I8", "off", 1);
    off.open(basename, "i8t");
    unsetenv("QLEVER_VECTOR_SEARCH_I8");
    query.resize(kDim);
    std::mt19937 qrng(777);
    for (auto& x : query) {
      x = g(qrng);
    }
  }
};

I8Fixture& fixture() {
  static I8Fixture f;
  return f;
}

// The unified VNNI fast path vs the punned-metric baseline: same entities in
// the same order, distances within the documented ~1 ulp (<= 1e-6). k=600
// takes the histogram route on `on`; `off` has no fast path (heap route).
TEST(VectorI8Scan, coarseTopKMatchesPunnedBaseline) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  const size_t k = 600;
  const auto got = f.on.searchExactCoarse(f.query, k);
  const auto ref = f.off.searchExactCoarse(f.query, k);
  ASSERT_EQ(got.size(), ref.size());
  ASSERT_EQ(got.size(), k);
  for (size_t i = 0; i < k; ++i) {
    EXPECT_EQ(got[i].entity_, ref[i].entity_) << i;
    EXPECT_NEAR(got[i].distance_, ref[i].distance_, 1e-6) << i;
  }
}

// The histogram route (k >= 512) and the blocked-sweep + heap route (k < 512)
// must agree exactly: the top-100 is a prefix of the top-600.
TEST(VectorI8Scan, histogramAndHeapRoutesAgree) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  const auto viaHeap = f.on.searchExactCoarse(f.query, 100);
  const auto viaHist = f.on.searchExactCoarse(f.query, 600);
  ASSERT_EQ(viaHeap.size(), 100u);
  ASSERT_GE(viaHist.size(), 100u);
  for (size_t i = 0; i < viaHeap.size(); ++i) {
    EXPECT_EQ(viaHeap[i].entity_, viaHist[i].entity_) << i;
    EXPECT_EQ(viaHeap[i].distance_, viaHist[i].distance_) << i;
  }
}

// ROUTING regression: the whole-index scan log line must NAME the route the
// top-k takes, and the i8 histogram gate (`k >= I8_HISTOGRAM_MIN_K = 512` on
// a tombstone-free VNNI store) must fire for a large k -- the production
// two-layer rerankK shape. Without the route in the log, a silent fall-back
// to a per-row engine (~2.4x slower at the same VNNI capability) is
// indistinguishable from the fast path in production logs.
TEST(VectorI8Scan, wholeIndexScanLogNamesTheRouteTaken) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  // k >= 512: the O(n) float-histogram select.
  testing::internal::CaptureStdout();
  (void)f.on.searchExactCoarse(f.query, 600);
  std::string log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("i8 block sweep + histogram select"), std::string::npos)
      << log;
  // k < 512: the blocked VNNI sweep + bounded heap.
  testing::internal::CaptureStdout();
  (void)f.on.searchExactCoarse(f.query, 100);
  log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("i8 block sweep + heap"), std::string::npos) << log;
  // The `Punned` dial keeps the per-row sidecar engine (still exact).
  testing::internal::CaptureStdout();
  (void)f.on.searchExactCoarse(f.query, 600, std::nullopt, std::nullopt, {},
                               nullptr, I8Kernel::Punned);
  log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("i8 per-row sweep + heap"), std::string::npos) << log;
  // The disabled fast path (QLEVER_VECTOR_SEARCH_I8=off at open) has no i8
  // kernel at all -> the usearch punned metric.
  testing::internal::CaptureStdout();
  (void)f.off.searchExactCoarse(f.query, 600);
  log = testing::internal::GetCapturedStdout();
  EXPECT_NE(log.find("punned per-row sweep + heap"), std::string::npos) << log;
}

// The `vec:i8Kernel` dial is a pure performance A/B on a VNNI CPU: `Punned`
// (per-row engine) and `Auto` (block engine) return BIT-IDENTICAL results
// (both compute the one shared integer-dot + finalize).
TEST(VectorI8Scan, punnedDialIsBitIdenticalToAuto) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  for (size_t k : {100u, 600u}) {
    const auto autoK = f.on.searchExactCoarse(
        f.query, k, std::nullopt, std::nullopt, {}, nullptr, I8Kernel::Auto);
    const auto punned = f.on.searchExactCoarse(
        f.query, k, std::nullopt, std::nullopt, {}, nullptr, I8Kernel::Punned);
    ASSERT_EQ(autoK.size(), punned.size()) << k;
    for (size_t i = 0; i < autoK.size(); ++i) {
      EXPECT_EQ(autoK[i].entity_, punned[i].entity_) << k << " " << i;
      EXPECT_EQ(autoK[i].distance_, punned[i].distance_) << k << " " << i;
    }
  }
}

// A NEAR-covering candidate set (every member but one non-top row) takes the
// scattered VNNI gather, and must return the identical coarse top-k.
TEST(VectorI8Scan, filteredCandidatesMatchWholeIndex) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  const size_t k = 50;
  const auto whole = f.on.searchExactCoarse(f.query, k);
  std::vector<Id> all(f.on.numLiveVectors());
  f.on.memberEntities(all);
  auto inTop = [&](Id id) {
    return std::any_of(whole.begin(), whole.end(),
                       [&](const ScoredEntity& e) { return e.entity_ == id; });
  };
  size_t excluded = 0;
  while (excluded < all.size() && inTop(all[excluded])) {
    ++excluded;
  }
  ASSERT_LT(excluded, all.size());
  std::vector<Id> nearCover;
  nearCover.reserve(all.size() - 1);
  for (size_t i = 0; i < all.size(); ++i) {
    if (i != excluded) {
      nearCover.push_back(all[i]);
    }
  }
  const auto filtered =
      f.on.searchExactCoarse(f.query, k, ql::span<const Id>{nearCover});
  ASSERT_EQ(whole.size(), filtered.size());
  for (size_t i = 0; i < whole.size(); ++i) {
    EXPECT_EQ(whole[i].entity_, filtered[i].entity_) << i;
    EXPECT_EQ(whole[i].distance_, filtered[i].distance_) << i;
  }
}

// Coarse-with-rows + fine rerank over ALL rows recovers the exact fine top-k
// (recall 1 at rerankK = n), and the rows the coarse pass reports align with
// its entities.
TEST(VectorI8Scan, coarsePlusRerankRecoversExactTopK) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  const size_t k = 10;
  const auto exact = f.on.searchExact(f.query, k);  // fine (bf16) baseline
  const auto coarse = f.on.searchExactCoarseWithRows(f.query, I8Fixture::kN);
  ASSERT_EQ(coarse.size(), I8Fixture::kN);
  const auto reranked =
      f.on.searchExactByRows(f.query, k, ql::span<const ScoredRow>{coarse});
  ASSERT_EQ(exact.size(), reranked.size());
  for (size_t i = 0; i < exact.size(); ++i) {
    EXPECT_EQ(exact[i].entity_, reranked[i].entity_) << i;
    // Both are FINE (bf16) distances, but through different engines: at this
    // fixture's dim=100 (not a multiple of 32) the whole-index sweep uses the
    // ragged-depth AMX GEMM while the by-rows rerank scores per pair -- the
    // documented ~1e-6 cross-kernel tolerance of the bf16 layer applies
    // (pre-existing bf16 behavior, unrelated to the i8 coarse pass).
    EXPECT_NEAR(exact[i].distance_, reranked[i].distance_, 1e-6) << i;
  }
}

// The CSLS coarse pass on an i8 scan layer: with the fast path the coarse
// select runs through the blocked VNNI sweep + the O(n) float-histogram
// select; the `Punned` dial keeps the per-row engine + `CoarseSelector`
// heap. Both must return IDENTICAL survivors (same shared distances, and
// the two selects are bit-identical); the punned-metric baseline
// (QLEVER_VECTOR_SEARCH_I8=off) must agree on the survivor set.
TEST(VectorI8Scan, cslsCoarseHistogramMatchesHeapAndBaseline) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  // A dedicated csls-enabled fixture (the shared one has no `.csls` sidecar).
  const std::string dir =
      (std::filesystem::temp_directory_path() / "qlever-i8scan-csls-test")
          .string();
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  const std::string basename = dir + "/idx";
  constexpr size_t kN = 3000;
  constexpr size_t kDim = 100;
  VectorIndexConfig cfg;
  cfg.name_ = "i8c";
  cfg.dimensions_ = kDim;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = VectorScalar::I8;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  {
    VectorIndexBuilder builder{basename, cfg};
    std::mt19937 rng(4711);
    std::normal_distribution<float> g(0.f, 1.f);
    std::vector<float> row(kDim);
    for (size_t i = 0; i < kN; ++i) {
      for (auto& x : row) {
        x = g(rng);
      }
      builder.add(mkId(100 + i), "<http://ex/c" + std::to_string(i) + ">", row);
    }
    builder.build();
  }
  unsetenv("QLEVER_VECTOR_SEARCH_I8");
  VectorIndex on;
  on.open(basename, "i8c");
  setenv("QLEVER_VECTOR_SEARCH_I8", "off", 1);
  VectorIndex off;
  off.open(basename, "i8c");
  unsetenv("QLEVER_VECTOR_SEARCH_I8");
  // A floor well below n, so the histogram select genuinely PARTITIONS the
  // coarse ranking (m = 500 of 3000) instead of degenerating to keep-all.
  on.setCslsRerankFloor(500);
  off.setCslsRerankFloor(500);
  std::vector<float> query(kDim);
  std::mt19937 qrng(31337);
  std::normal_distribution<float> g(0.f, 1.f);
  for (auto& x : query) {
    x = g(qrng);
  }
  const float threshold = -0.5f;  // keeps a moderate survivor set
  const auto viaHist =
      on.searchCsls(query, threshold, 10, std::nullopt, std::nullopt, {},
                    nullptr, false, Bf16Kernel::Auto, I8Kernel::Auto);
  const auto viaHeap =
      on.searchCsls(query, threshold, 10, std::nullopt, std::nullopt, {},
                    nullptr, false, Bf16Kernel::Auto, I8Kernel::Punned);
  ASSERT_FALSE(viaHist.empty());
  // Same shared coarse distances + bit-identical selects -> identical output.
  ASSERT_EQ(viaHist.size(), viaHeap.size());
  for (size_t i = 0; i < viaHist.size(); ++i) {
    EXPECT_EQ(viaHist[i].entity_, viaHeap[i].entity_) << i;
    EXPECT_EQ(viaHist[i].distance_, viaHeap[i].distance_) << i;
    EXPECT_EQ(viaHist[i].csls_, viaHeap[i].csls_) << i;
  }
  // The punned-metric baseline agrees on the survivors (its ~1 ulp coarse
  // distances can only shift the coarse ordering, not this fixture's
  // clearly-separated survivor set; the FINE distances are identical).
  const auto baseline =
      off.searchCsls(query, threshold, 10, std::nullopt, std::nullopt, {},
                     nullptr, false, Bf16Kernel::Auto, I8Kernel::Auto);
  ASSERT_EQ(viaHist.size(), baseline.size());
  for (size_t i = 0; i < viaHist.size(); ++i) {
    EXPECT_EQ(viaHist[i].entity_, baseline[i].entity_) << i;
    EXPECT_EQ(viaHist[i].distance_, baseline[i].distance_) << i;
    EXPECT_NEAR(viaHist[i].csls_, baseline[i].csls_, 1e-5) << i;
  }
}

// `maxDistance` filters the OUTPUT of the histogram route exactly like the
// heap route's post-filter.
TEST(VectorI8Scan, maxDistanceFilterOnHistogramRoute) {
  if (!i8kernels::vnniAvailable()) {
    GTEST_SKIP() << "no AVX-512-VNNI on this CPU";
  }
  auto& f = fixture();
  const size_t k = 600;
  const auto unfiltered = f.on.searchExactCoarse(f.query, k);
  ASSERT_EQ(unfiltered.size(), k);
  // A threshold between the median and the max of the top-k distances.
  const float cutoff = unfiltered[k / 2].distance_;
  const auto filtered =
      f.on.searchExactCoarse(f.query, k, std::nullopt, cutoff);
  ASSERT_FALSE(filtered.empty());
  ASSERT_LT(filtered.size(), k);
  for (size_t i = 0; i < filtered.size(); ++i) {
    EXPECT_EQ(filtered[i].entity_, unfiltered[i].entity_) << i;
    EXPECT_LE(filtered[i].distance_, cutoff) << i;
  }
}

}  // namespace
