// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSELECT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSELECT_H

#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "services/vectorSearch/VectorSweepExecutor.h"

namespace qlever::vector {

// O(n) selection of the `m` smallest `(distance, index)` pairs -- ascending
// by the FULL pair order (distance, then index) -- from a materialized
// distance array, via a fixed-point HISTOGRAM instead of a bounded heap.
// BIT-IDENTICAL to `CoarseSelector` (per-thread `(distance, index)` heaps +
// merge) and to `nth_element(keyed, m) + sort(prefix)`: all three realise the
// same total pair order.
//
// This is the FLOAT-distance sibling of the integer counting select the
// binary (Hamming) coarse layer uses (`coarseSweepSelectHistogram`), for the
// i8 coarse layer's quantized-cosine distances in [0, 2]: a bounded heap at a
// production-size `m` (the i8 `rerankK` / `cslsRerankFloor`, up to 10^4)
// costs ~k*ln(n/k) cache-spilling heap pushes plus an O(threads*m) merge --
// measured ~8 ms extra on a 2.14M-row sweep at m=10^4 (24-core EMR Xeon),
// where this select costs ~1 ms and neither anti-scales with threads nor
// with `m`.
//
// HOW: distances are binned by a MONOTONIC fixed-point quantization
// (`bucket = min(NB-1, floor(d * NB/2))`, 4096 buckets over [0, 2]); the
// smallest bucket `tb` whose cumulative count reaches `m` bounds the
// selection (every one of the `m` smallest pairs has `d <=` some distance in
// `tb`, hence bucket `<= tb` by monotonicity). One collect pass gathers
// exactly the pairs in buckets `<= tb` -- `m` plus at most one bucket's
// overshoot -- and a final `nth_element + sort` over that SMALL set yields
// the exact (distance, index) prefix. Never worse than O(n + c log c) with
// `c` the collected count (== n only if all distances share one bucket, in
// which case the old heap was no better).
//
// PRECONDITION: every `dists[i]` is a finite, non-negative float (the i8
// angular finalize guarantees [0, ~2]; values above 2 merely share the last
// bucket). NaNs would break the sort's strict weak ordering -- the caller's
// kernel never produces them.
//
// `numThreads` > 1 parallelizes the two O(n) passes (binning + collect) over
// contiguous index ranges; the result is independent of the thread count
// (each pass writes disjoint state; the merge/sort is deterministic).
//
// `collectedCount` (optional out-param): the number of pairs the collect
// pass gathered, i.e. the `nth_element` input size -- `m` plus the boundary
// bucket's overshoot. Clustered distances (many rows sharing the boundary
// bucket) inflate it; callers log it so a pathological distribution is
// visible in production.
inline std::vector<std::pair<float, uint64_t>> selectSmallestPairsFloat(
    const float* dists, size_t n, size_t m, int numThreads,
    size_t* collectedCount = nullptr) {
  using Pair = std::pair<float, uint64_t>;
  if (collectedCount != nullptr) {
    *collectedCount = 0;
  }
  m = std::min(m, n);
  if (m == 0) {
    return {};
  }
  constexpr size_t NB = 4096;
  constexpr float SCALE = static_cast<float>(NB) / 2.0f;  // d in [0, 2]
  auto bucketOf = [](float d) -> size_t {
    const size_t b = static_cast<size_t>(d * SCALE);
    return b < NB ? b : NB - 1;
  };
#ifdef _OPENMP
  const size_t nt = numThreads > 1 ? static_cast<size_t>(numThreads) : 1;
#else
  const size_t nt = 1;
#endif
  // 1. Per-thread histograms over the fixed buckets (cache-resident,
  //    branchless increment), merged into `hist`. The per-thread counts are
  //    RETAINED so the collect pass can reserve exactly.
  std::vector<std::vector<uint32_t>> localHists(nt,
                                                std::vector<uint32_t>(NB, 0));
  auto binRange = [&](size_t t) {
    uint32_t* h = localHists[t].data();
    const size_t first = n * t / nt;
    const size_t last = n * (t + 1) / nt;
    for (size_t i = first; i < last; ++i) {
      ++h[bucketOf(dists[i])];
    }
  };
#ifdef _OPENMP
  if (nt > 1) {
    runVectorSweep({}, [&] {
#pragma omp parallel for schedule(static) num_threads(numThreads)
      for (size_t t = 0; t < nt; ++t) {
        binRange(t);
      }
    });
  } else
#endif
  {
    binRange(0);
  }
  std::vector<uint64_t> hist(NB, 0);
  for (const auto& lh : localHists) {
    for (size_t b = 0; b < NB; ++b) {
      hist[b] += lh[b];
    }
  }
  // 2. Boundary bucket `tb`: the smallest bucket whose cumulative count
  //    reaches `m`.
  size_t tb = NB - 1;
  {
    size_t cum = 0;
    for (size_t b = 0; b < NB; ++b) {
      cum += hist[b];
      if (cum >= m) {
        tb = b;
        break;
      }
    }
  }
  // 3. Collect every pair in buckets <= tb (a small superset of the m
  //    smallest), each thread into an exactly-reserved local part.
  std::vector<std::vector<Pair>> parts(nt);
  auto collectRange = [&](size_t t) {
    size_t mine = 0;
    for (size_t b = 0; b <= tb; ++b) {
      mine += localHists[t][b];
    }
    auto& out = parts[t];
    out.reserve(mine);
    const size_t first = n * t / nt;
    const size_t last = n * (t + 1) / nt;
    for (size_t i = first; i < last; ++i) {
      const float d = dists[i];
      if (bucketOf(d) <= tb) {
        out.emplace_back(d, static_cast<uint64_t>(i));
      }
    }
  };
#ifdef _OPENMP
  if (nt > 1) {
    runVectorSweep({}, [&] {
#pragma omp parallel for schedule(static) num_threads(numThreads)
      for (size_t t = 0; t < nt; ++t) {
        collectRange(t);
      }
    });
  } else
#endif
  {
    collectRange(0);
  }
  // 4. Concatenate (thread parts are index-ascending and disjoint) and take
  //    the exact m smallest by the full (distance, index) pair order.
  std::vector<Pair> all;
  {
    size_t total = 0;
    for (const auto& p : parts) {
      total += p.size();
    }
    all.reserve(total);
    for (const auto& p : parts) {
      all.insert(all.end(), p.begin(), p.end());
    }
  }
  if (collectedCount != nullptr) {
    *collectedCount = all.size();
  }
  if (all.size() > m) {
    std::nth_element(all.begin(), all.begin() + static_cast<std::ptrdiff_t>(m),
                     all.end());
    all.resize(m);
  }
  std::sort(all.begin(), all.end());
  return all;
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSELECT_H
