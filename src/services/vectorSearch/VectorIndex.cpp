// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndex.h"

#include <sys/mman.h>

#include <algorithm>
#include <cctype>
#include <iomanip>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <string_view>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "backports/StartsWithAndEndsWith.h"
#include "backports/algorithm.h"
#include "services/vectorSearch/UsearchGraph.h"
#include "services/vectorSearch/VectorMemory.h"
#include "util/Exception.h"
#include "util/HashSet.h"
#include "util/Log.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
// How often the exact-search loops call the interrupt callback. Frequent enough
// for sub-second cancellation latency, rare enough to not show up in profiles.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// TEMPORARY DIAGNOSTIC (normally true): when false, the per-row cancellation
// polling is compiled OUT of the hot PARALLEL scan loops (the `&&` short-circuits
// at compile time, so even the modulo is gone) to measure its cost. Queries can
// then NOT be interrupted mid-scan. Flip back to true before merging.
constexpr bool VEC_SEARCH_CHECK_CANCELLATION = false;

// Hard upper bound on the number of results a single search returns, regardless
// of the user's `k` and the index size. `k` is remote and unbounded; without
// this cap a query like `vec:k 100000000` on a 100M-vector index would drive
// gigabytes of (unaccounted) scratch and an effectively whole-index HNSW probe.
// A nearest-neighbour query asking for more than this is pathological; the
// result is capped (not an error). 100k results = ~1.2 MB of key+distance
// buffers, and a bounded HNSW search.
constexpr size_t MAX_SEARCH_RESULTS = 100'000;

// A pool of usearch search-context ids. Each concurrent HNSW search needs a
// distinct context; a request beyond the pool size briefly WAITS for a free
// context instead of failing the query (the pool is sized generously above the
// hardware concurrency, so waiting is rare).
class SearchSlotPool {
 public:
  explicit SearchSlotPool(size_t numSlots) {
    for (size_t i = 0; i < numSlots; ++i) {
      free_.push(i);
    }
  }

  class Slot {
   public:
    Slot(SearchSlotPool& pool, size_t id) : pool_{&pool}, id_{id} {}
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() { pool_->release(id_); }
    size_t id() const { return id_; }

   private:
    SearchSlotPool* pool_;
    size_t id_;
  };

  // Acquire a slot, waiting if none is free. `checkInterrupt`, if set, is
  // invoked while waiting so that a cancelled/timed-out query does not park a
  // worker thread here indefinitely (it throws to unwind).
  Slot acquire(const CheckInterruptCallback& checkInterrupt) {
    std::unique_lock lock{mutex_};
    while (free_.empty()) {
      if (checkInterrupt) {
        cv_.wait_for(lock, std::chrono::milliseconds{50},
                     [this] { return !free_.empty(); });
        if (free_.empty()) {
          lock.unlock();
          checkInterrupt();  // throws on cancellation
          lock.lock();
        }
      } else {
        cv_.wait(lock, [this] { return !free_.empty(); });
      }
    }
    size_t id = free_.front();
    free_.pop();
    return Slot{*this, id};
  }

 private:
  void release(size_t id) {
    {
      std::lock_guard lock{mutex_};
      free_.push(id);
    }
    cv_.notify_one();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<size_t> free_;
};
}  // namespace

// Deleter for the `posix_memalign`-allocated aligned copy (freed with
// `std::free`, NOT `delete`).
struct AlignedFree {
  void operator()(void* p) const { std::free(p); }
};

// All heavy members (mmaped files + usearch) live here so that neither usearch
// nor `MmapVector` leak into the public header.
struct VectorIndex::Impl {
  // One flat row-major matrix with its metric, f32 casts, and RAM-residency
  // state. Every index has the SCAN layer (`.data`, the configured `scalar_`);
  // a two-layer index additionally has the fine RERANK layer (`.rerank.data`).
  // Both share the entity mappings (`keys_`/`rowmap_`) and the row order.
  struct MatrixLayer {
    ad_utility::MmapVectorView<char> data_;
    std::optional<uu::metric_punned_t> metric_;
    uu::casts_punned_t casts_;  // f32 <-> this layer's storage scalar
    size_t dim_ = 0;
    // Raw (unpadded) per-row byte length of this layer.
    size_t rowBytes_ = 0;
    // Byte stride between consecutive rows (>= `rowBytes_`). Derived from the
    // metadata at open; for a `Residency::AlignedCopy` it becomes the natural
    // (compact) stride of `alignedBuf_`.
    size_t stride_ = 0;
    // Optional 64-byte-aligned, hugepage-advised RAM copy of the whole matrix
    // (`Residency::AlignedCopy`), rows at the natural stride. When set,
    // `base()` reads from it instead of the memory-mapped file.
    std::unique_ptr<char, AlignedFree> alignedBuf_;
    // The residency strategy last applied (best-effort; see
    // `VectorIndex::residency()`). Stays `None` when the request was skipped
    // by the fits-in-RAM gate.
    Residency residency_ = Residency::None;

#if USEARCH_USE_NUMKONG
    // P3: the RAW NumKong per-pair distance kernel for this layer, resolved
    // ONCE at open via `nk_find_kernel_punned` -- the SAME kernel usearch's
    // `metric_punned_t` bakes in, so results are bit-identical, but callable
    // directly (one indirection, no `optional::value()` / member-function-
    // pointer bit test / result-through-stack of the punned wrapper). The hot
    // sweeps call this instead of `distanceBetweenBytes` where profitable.
    // `nullptr` on a non-NumKong build or an unresolvable kernel -> callers
    // fall back to the punned `metric_`.
    //
    // Signature is the NumKong dense-metric ABI `(a, b, n, &result)`; the
    // result scalar width depends on the kernel (u32 for Hamming, f32 for
    // angular/bf16), so it is stored per-typed-invoker below.
    using RawU32Kernel = void (*)(const void*, const void*, nk_size_t,
                                  nk_u32_t*);
    RawU32Kernel rawHammingKernel_ = nullptr;
    // The third argument of the raw kernel: for the u1 (binary) Hamming kernel
    // this is the number of BITS (`dim_`), NOT bytes -- exactly what usearch
    // passes (`index_plugins.hpp`: `metric_third_arg_ = dimensions_` for
    // b1x8). Only meaningful when `rawHammingKernel_ != nullptr`.
    nk_size_t rawKernelThirdArg_ = 0;

    // P2: batched bf16 cosine path. Present iff this is a bf16 layer with a
    // Cosine metric AND the running CPU has the AMX (sapphireamx) batched
    // GEMM -- the ONLY batched backend measured faster than the per-row kernel
    // at the width=1 query shape (the AVX-512-BF16 GEMM regresses; verified).
    // When set, the fine sweeps compute a whole BLOCK of cosine distances with
    // one `nk_dots_packed_bf16` (query x contiguous rows) plus a per-row norm
    // finalize, instead of one `nk_angular_bf16` call per row. `nullptr`
    // packed buffer -> not eligible -> callers use the per-row path.
    bool batchedCosineBf16_ = false;
    // One `1 / sqrt(sumsq(row))` per row (`numVectors` entries), computed in
    // one parallel pass at open (~2 B/dim, e.g. 8.6 MB at 2.14M rows). Lets
    // the batched finalize turn a raw dot product into a cosine distance
    // without re-reading the (bf16) row. Empty unless `batchedCosineBf16_`.
    std::vector<float> invNorms_;
#endif

    // Base pointer of the active matrix: the aligned RAM copy if present,
    // else the memory-mapped file.
    const char* base() const {
      return alignedBuf_ ? alignedBuf_.get() : data_.data();
    }

    // Pointer to the start of row `i` in the active matrix (storage scalar
    // bytes). Uses the per-row stride, which may exceed `rowBytes_` when rows
    // are padded for SIMD alignment; the metric reads only `dim_` scalars.
    const char* rowPtr(size_t i) const { return base() + i * stride_; }

    // Encode an f32 query into this layer's storage scalar. Returns a pointer
    // to the encoded bytes; `buffer` provides the storage when a conversion
    // happens. The buffer is ZEROED first (`assign`, not `resize`): usearch's
    // f32 -> b1x8 sign-pack cast memsets only the `dim / 8` WHOLE output
    // bytes and then ORs bits in, so a partial trailing byte of a reused
    // buffer would keep stale bits (all other casts overwrite every byte, for
    // which the zeroing is merely redundant).
    const char* encodeQuery(ql::span<const float> query,
                            std::vector<char>& buffer) const {
      const char* raw = reinterpret_cast<const char*>(query.data());
      auto fromF32 = casts_.from.f32;
      if (fromF32 != nullptr) {
        buffer.assign(rowBytes_, 0);
        if (fromF32(raw, dim_, buffer.data())) {
          return buffer.data();
        }
      }
      return raw;
    }

    // Distance between two (encoded, `rowBytes_`-sized) vectors.
    //
    // For the bf16 cosine fine layer with the batched path enabled (P2), this
    // uses the SAME dot-product-plus-norm-sidecar cosine as the batched block
    // sweep, so every exact bf16 distance -- whole-index sweep, CSLS sweep,
    // rerank batch, and the single-pair `vec:distance` API -- agrees on any
    // given pair to the bit. It differs from usearch's internal (HNSW) metric
    // by at most ~1e-6 (the batched kernel rounds the norm terms differently);
    // that tolerance is accepted (`vec:distance` is the exact baseline, the
    // HNSW graph only pre-selects candidates that are then re-scored exactly).
    // Every other layer/metric uses the (punned) usearch metric directly, so
    // exact and HNSW distances stay identical there.
    float distanceBetweenBytes(const char* a, const char* b) const {
#if USEARCH_USE_NUMKONG
      if (batchedCosineBf16_) {
        return cosineBf16(a, b, queryInvNormBf16(a, dim_),
                          queryInvNormBf16(b, dim_));
      }
#endif
      return static_cast<float>(
          metric_.value()(reinterpret_cast<const uu::byte_t*>(a),
                          reinterpret_cast<const uu::byte_t*>(b)));
    }

    // Distance between an (encoded) query and row `i`.
    float distanceToRow(const char* queryBytes, size_t i) const {
#if USEARCH_USE_NUMKONG
      if (batchedCosineBf16_) {
        return cosineToRowBf16(queryBytes, i, queryInvNormBf16(queryBytes, dim_));
      }
#endif
      return distanceBetweenBytes(queryBytes, rowPtr(i));
    }

#if USEARCH_USE_NUMKONG
    // P3: Hamming distance via the raw kernel resolved at open (see
    // `rawHammingKernel_`). `queryBytes`/`b` are `rowBytes_`-sized encoded
    // vectors. Only valid when `rawHammingKernel_ != nullptr`; the u1 kernel
    // takes the BIT count (`rawKernelThirdArg_`), not the byte count.
    uint32_t rawHamming(const char* queryBytes, const char* b) const {
      nk_u32_t r;
      rawHammingKernel_(queryBytes, b, rawKernelThirdArg_, &r);
      return static_cast<uint32_t>(r);
    }

    // P2: `sumsq(row)` over a bf16-encoded vector -- each bf16 is the high 16
    // bits of an f32, so decode by `<< 16` and accumulate the squares in f32.
    // Used to build the norm sidecar and the per-query reciprocal norm; kept
    // dependency-free (no NumKong internal helper) and deterministic.
    static float sumsqBf16(const char* bf16, size_t dim) {
      const auto* h = reinterpret_cast<const uint16_t*>(bf16);
      float sumsq = 0.0f;
      for (size_t d = 0; d < dim; ++d) {
        const uint32_t bits = static_cast<uint32_t>(h[d]) << 16;
        float v;
        std::memcpy(&v, &bits, sizeof(v));
        sumsq += v * v;
      }
      return sumsq;
    }

    // P2: the reciprocal query norm `1 / sqrt(sumsq(query))` for the batched
    // bf16 cosine finalize. `queryBf16` is the query already encoded to this
    // layer's bf16 storage (`encodeQuery`).
    static float queryInvNormBf16(const char* queryBf16, size_t dim) {
      const float sumsq = sumsqBf16(queryBf16, dim);
      // rsqrt(0) would be +inf; a zero query yields cosine distance 1 for all
      // rows (dot is 0), matching `1 - 0` -- clamp the norm term to 0 instead
      // of propagating inf * 0 = NaN.
      return sumsq > 0.0f ? 1.0f / std::sqrt(sumsq) : 0.0f;
    }

    // P2: pack the bf16 query ONCE per search into the opaque NumKong GEMM
    // layout (width=1). The caller keeps the returned buffer alive and passes
    // `.data()` to every `cosineBlockBf16` call. Only valid when
    // `batchedCosineBf16_` is true.
    std::vector<char> packQueryBf16(const char* queryBf16) const {
      std::vector<char> buf(nk_dots_packed_size_bf16(1, dim_));
      nk_dots_pack_bf16(reinterpret_cast<const nk_bf16_t*>(queryBf16),
                        /*width=*/1, dim_, rowBytes_, buf.data());
      return buf;
    }

    // P2: the shared cosine-from-dot finalize -- the ONE formula every exact
    // bf16 path uses, matching the sapphireamx angular kernel:
    //   d = max(0, 1 - dot * invNormA * invNormB).
    static float cosineFromDot(float dot, float invNormA, float invNormB) {
      const float d = 1.0f - dot * invNormA * invNormB;
      return d > 0.0f ? d : 0.0f;
    }

    // P2: cosine distance between two bf16-encoded vectors, computing both
    // norms live (the single-pair `vec:distance` slow path -- no sidecar row).
    float cosineBf16(const char* a, const char* b, float invNormA,
                     float invNormB) const {
      nk_f32_t dot;
      nk_dot_bf16(reinterpret_cast<const nk_bf16_t*>(a),
                  reinterpret_cast<const nk_bf16_t*>(b), dim_, &dot);
      return cosineFromDot(static_cast<float>(dot), invNormA, invNormB);
    }

    // P2: cosine distance between a bf16 query (reciprocal norm `queryInvNorm`
    // pre-computed once) and store row `i`, reading the row's reciprocal norm
    // from the sidecar. The per-row fallback used by the SCATTERED (rerank
    // gather) and non-AMX fine sweeps; bit-identical to the batched block for
    // the same pair (same dot kernel family, same finalize).
    float cosineToRowBf16(const char* queryBf16, size_t i,
                          float queryInvNorm) const {
      nk_f32_t dot;
      nk_dot_bf16(reinterpret_cast<const nk_bf16_t*>(queryBf16),
                  reinterpret_cast<const nk_bf16_t*>(rowPtr(i)), dim_, &dot);
      return cosineFromDot(static_cast<float>(dot), queryInvNorm, invNorms_[i]);
    }

    // P2: cosine distances for a CONTIGUOUS block of `count` rows starting at
    // store row `rowStart`, via one AMX batched dot product (`packedQuery`
    // from `packQueryBf16`) plus a per-row norm finalize. `queryInvNorm` is
    // the query's reciprocal norm (`queryInvNormBf16`), `dots` a
    // caller-provided scratch buffer of >= `count` floats, `out[j]` receives
    // the cosine distance of row `rowStart + j`. Only valid when
    // `batchedCosineBf16_` is true.
    void cosineBlockBf16(size_t rowStart, size_t count, const void* packedQuery,
                         float queryInvNorm, float* dots, float* out) const {
      const char* rows = base() + rowStart * stride_;
      nk_dots_packed_bf16(reinterpret_cast<const nk_bf16_t*>(rows), packedQuery,
                          dots, count, /*width=*/1, dim_, stride_,
                          /*c_stride=*/sizeof(float));
      const float* invN = invNorms_.data() + rowStart;
      for (size_t j = 0; j < count; ++j) {
        out[j] = cosineFromDot(dots[j], invN[j], queryInvNorm);
      }
    }
#endif
  };

  VectorIndexMetadata meta_;
  ad_utility::MmapVectorView<uint64_t> keys_;  // row -> id (or TOMBSTONE_KEY)
  ad_utility::MmapVectorView<IdRowPair> rowmap_;  // id -> row, sorted by id
  // The coarse scan matrix `.data` (the only matrix of a single-layer index).
  MatrixLayer scan_;
  // The fine rerank matrix `.rerank.data`, iff `meta_.config_.rerankScalar_`.
  std::optional<MatrixLayer> rerank_;
  // The row-aligned csls hubness sidecar `.csls` (one f32 r(d) per row), iff
  // `meta_.config_.csls_`.
  ad_utility::MmapVectorView<float> cslsR_;
  // Fine-rerank batch size M of the two-layer CSLS cut (see `searchCslsBytes`:
  // coarse-scan everything, rerank only the coarse-best M, widen by M while
  // the cut still reaches the coarse boundary). A pure SERVING setting --
  // never persisted; overridable per index at server start via the
  // `cslsRerankFloor` key of `QLEVER_VECTOR_SEARCH_ENDPOINTS`.
  size_t cslsRerankFloor_ = DEFAULT_CSLS_RERANK_FLOOR;
  // Per-index serving DEFAULTS of the dynamic `vec:autoCut` cuts (query
  // params override them, unset falls back to the `DEFAULT_CSLS_*`
  // constants; see `resolveCslsCut`). Like `cslsRerankFloor_`: never
  // persisted, reapplied by the load hook from
  // `QLEVER_VECTOR_SEARCH_ENDPOINTS`.
  std::optional<float> cslsFloorDefault_;
  std::optional<float> softmaxTemperatureDefault_;
  std::optional<size_t> softmaxNDefault_;
  std::optional<float> breadthDefault_;
  // Top-anchored z-cut serving defaults: band fractions [precise, balanced,
  // broad], the exact no-match gate, and the floor-estimator fraction.
  std::array<std::optional<float>, 3> zcutFractionDefault_;
  std::optional<float> zcutGateZDefault_;
  std::optional<float> zcutFloorFractionDefault_;
  std::optional<GraphIndex> graph_;  // present iff meta_.hasHnsw_
  std::unique_ptr<SearchSlotPool> searchSlots_;
  // Sticky safety net for the sequential-read hint of the merge-join gather.
  // The gather emits rows in non-decreasing order iff the store is genuinely
  // id-sorted (the normal case). If a gather is ever observed to emit rows out
  // of order (a changed KG collation, so the store's layout no longer matches
  // the current ids), this latches and later gathers stop advising SEQUENTIAL
  // (which would mislead the kernel's read-ahead). Correctness is unaffected
  // either way; this only tunes the `madvise` hint. `mutable` so the const
  // search methods can set it; `atomic` for concurrent searches.
  mutable std::atomic<bool> gatherNonMonotonic_{false};

  size_t dim() const { return meta_.config_.dimensions_; }
  size_t numLive() const { return meta_.numVectors_ - meta_.numTombstones_; }

  // The FINE layer: what the exact primitives (`vec:distance`,
  // `searchExact*`, `getVector`) score on -- the rerank matrix when present,
  // else the single scan store. (The non-const overload exists solely for the
  // advisory `madvise` hints of the exact scans.)
  const MatrixLayer& fine() const {
    return rerank_.has_value() ? rerank_.value() : scan_;
  }
  MatrixLayer& fine() { return rerank_.has_value() ? rerank_.value() : scan_; }

  // Row index of `entity`, or nullopt if it has no (live) vector here.
  std::optional<size_t> rowOf(Id entity) const {
    uint64_t id = entity.getBits();
    auto it = ql::ranges::lower_bound(
        rowmap_, id, std::less<>{},
        [](const IdRowPair& pair) { return pair.idBits_; });
    if (it == rowmap_.end() || it->idBits_ != id) {
      return std::nullopt;
    }
    return static_cast<size_t>(it->row_);
  }

  // The HNSW graph reads the SCAN layer (the graph is built on it, so its
  // distances live in the coarse space of a two-layer index).
  FlatStoreMetric graphMetric() const {
    return FlatStoreMetric{scan_.base(), scan_.stride_, meta_.numVectors_,
                           scan_.metric_.value()};
  }
};

// Map a `preload` string to a `Residency`. An unknown value (should not
// happen -- validated at build time and by the runtime-override parser) falls
// back to `None`.
VectorIndex::Residency VectorIndex::residencyFromString(const std::string& s) {
  if (s == "advise") return Residency::Advise;
  if (s == "lock") return Residency::Lock;
  if (s == "aligned") return Residency::AlignedCopy;
  return Residency::None;
}

// Forward declaration (defined below `open`): apply a RAM-residency strategy
// to one matrix layer. (A template so that it can take the private
// `VectorIndex::Impl::MatrixLayer` without naming it.)
template <typename LayerT>
void makeLayerResident(LayerT& layer, VectorIndex::Residency residency,
                       size_t numRows, const std::string& indexName,
                       const char* layerLabel);

// Resolve the NumKong fast paths of one matrix layer at open (P2/P3):
//
//  - P3 (any layer with an integer Hamming metric, i.e. the binary SCAN
//    layer): resolve the RAW per-pair Hamming kernel once so the hot coarse
//    sweep can call it directly instead of through `metric_punned_t`. It is
//    the same kernel usearch baked into the punned metric.
//  - P2 (a bf16 fine layer with a Cosine metric, ONLY when the CPU has the
//    AMX batched GEMM): enable the batched cosine sweep and build the per-row
//    `1/sqrt(sumsq)` norm sidecar in one parallel pass. On a non-AMX CPU the
//    batched GEMM at width=1 is a measured regression, so the layer keeps the
//    per-row kernel (and the exact bf16 distance stays the usearch metric).
//
// A no-op without NumKong. `scalar`/`metric` are the layer's storage scalar
// and metric; `numRows` the store row count; `numThreads` the sidecar-build
// fan-out.
template <typename LayerT>
void configureLayerFastPaths([[maybe_unused]] LayerT& layer,
                             [[maybe_unused]] VectorScalar scalar,
                             [[maybe_unused]] VectorMetric metric,
                             [[maybe_unused]] size_t numRows,
                             [[maybe_unused]] int numThreads) {
#if USEARCH_USE_NUMKONG
  const nk_capability_t caps = nk_capabilities_available();

  // P3: raw Hamming kernel for the binary (b1x8) scan layer.
  if (scalar == VectorScalar::Binary) {
    nk_kernel_punned_t kfn = nullptr;
    nk_capability_t used{};
    nk_find_kernel_punned(nk_kernel_hamming_k, nk_u1_k, caps, &kfn, &used);
    // memcpy the opaque `(void*)` kernel pointer into the typed slot (same
    // trick usearch uses for `metric_ptr_`): a plain reinterpret_cast between
    // incompatible function-pointer types trips -Wcast-function-type.
    typename LayerT::RawU32Kernel typed = nullptr;
    std::memcpy(&typed, &kfn, sizeof(typed));
    layer.rawHammingKernel_ = typed;
    // The u1 kernel's third argument is the number of BITS (== dim), not
    // bytes -- exactly what usearch passes for a b1x8 metric.
    layer.rawKernelThirdArg_ = static_cast<nk_size_t>(layer.dim_);
  }

  // P2: batched cosine for the bf16 fine layer, only with AMX.
  if (scalar == VectorScalar::Bf16 && metric == VectorMetric::Cosine &&
      (caps & nk_cap_sapphireamx_k) != 0) {
    layer.batchedCosineBf16_ = true;
    layer.invNorms_.assign(numRows, 0.0f);
    float* invN = layer.invNorms_.data();
    const size_t dim = layer.dim_;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(numThreads)
#endif
    for (size_t i = 0; i < numRows; ++i) {
      const float sumsq = LayerT::sumsqBf16(layer.rowPtr(i), dim);
      invN[i] = sumsq > 0.0f ? 1.0f / std::sqrt(sumsq) : 0.0f;
    }
  }
#endif
}

// ____________________________________________________________________________
VectorIndex::VectorIndex() : impl_{std::make_unique<Impl>()} {}
VectorIndex::~VectorIndex() = default;
VectorIndex::VectorIndex(VectorIndex&&) noexcept = default;
VectorIndex& VectorIndex::operator=(VectorIndex&&) noexcept = default;

// ____________________________________________________________________________
void VectorIndex::open(const std::string& basename, const std::string& name,
                       Residency residency, Residency rerankResidency) {
  auto& impl = *impl_;

  // 1. Metadata.
  std::ifstream metaIn{vectorMetaFile(basename, name)};
  if (!metaIn.is_open()) {
    AD_THROW("Could not open the metadata file of vector index \"" + name +
             "\": " + vectorMetaFile(basename, name));
  }
  nlohmann::json j;
  try {
    metaIn >> j;
    impl.meta_ = j.get<VectorIndexMetadata>();
  } catch (const std::exception& e) {
    AD_THROW("The metadata file of vector index \"" + name + "\" (" +
             vectorMetaFile(basename, name) + ") is malformed: " + e.what());
  }
  if (!isSupportedVectorIndexVersion(impl.meta_.version_)) {
    AD_THROW("Vector index \"" + name + "\" has on-disk version " +
             std::to_string(impl.meta_.version_) +
             ", but this binary expects version " +
             std::to_string(VECTOR_INDEX_VERSION) +
             " (or the still-supported version 4). Please rebuild the vector "
             "index.");
  }

  // 2. Flat store + entity mappings (memory-mapped).
  auto openMmap = [&](auto& view, const std::string& path,
                      ad_utility::AccessPattern pattern) {
    try {
      view.open(path, pattern);
    } catch (const std::exception& e) {
      AD_THROW("Could not open the file " + path + " of vector index \"" +
               name + "\": " + e.what());
    }
  };
  openMmap(impl.keys_, vectorKeysFile(basename, name),
           ad_utility::AccessPattern::Random);
  openMmap(impl.rowmap_, vectorRowmapFile(basename, name),
           ad_utility::AccessPattern::Random);
  openMmap(impl.scan_.data_, vectorDataFile(basename, name),
           ad_utility::AccessPattern::Random);
  auto complainInterrupted = [&](const std::string& what) {
    AD_THROW("Vector index \"" + name + "\": " + what +
             " does not match the metadata. The index files are likely from "
             "an interrupted build; please rebuild the vector index.");
  };
  if (impl.keys_.size() != impl.meta_.numVectors_) {
    complainInterrupted("the number of keys on disk");
  }
  if (impl.meta_.numTombstones_ > impl.meta_.numVectors_ ||
      impl.rowmap_.size() != impl.numLive()) {
    complainInterrupted("the entity mapping on disk");
  }
  if (impl.meta_.config_.dimensions_ == 0) {
    complainInterrupted("the dimension");
  }
  impl.scan_.dim_ = impl.dim();
  // Raw row byte length via `rowBytesFor`: `dim * bytesPerScalar`, except for
  // the sign-packed `binary` scalar whose rows are `(dim + 7) / 8` bytes.
  impl.scan_.rowBytes_ = rowBytesFor(impl.meta_.config_.scalar_, impl.dim());
  // Row stride: explicit for a v5 index, derived (== raw row byte length) for a
  // v4 index (`rowStrideBytes_ == 0`). It must be at least the raw row length
  // (the metric reads `rowBytes_` per row), or the mapped reads could go out
  // of bounds; the padded tail beyond it is never read.
  impl.scan_.stride_ = impl.meta_.rowStrideBytes_ != 0
                           ? impl.meta_.rowStrideBytes_
                           : impl.scan_.rowBytes_;
  if (impl.scan_.stride_ < impl.scan_.rowBytes_) {
    complainInterrupted("the row stride");
  }
  if (impl.scan_.data_.size() != impl.meta_.numVectors_ * impl.scan_.stride_) {
    complainInterrupted("the data size on disk");
  }
  // Validate the VALUES in the mapping (not just the counts): `.rowmap` must be
  // sorted by id and every `row_` must be in range, so that the unchecked
  // memory-mapped reads on the query paths can never go out of bounds on a
  // corrupt or truncated index file. One O(numLive) scan at load.
  for (size_t i = 0; i < impl.rowmap_.size(); ++i) {
    const IdRowPair& pair = impl.rowmap_[i];
    if (pair.row_ >= impl.meta_.numVectors_ || pair.idBits_ == TOMBSTONE_KEY ||
        (i > 0 && impl.rowmap_[i - 1].idBits_ >= pair.idBits_)) {
      complainInterrupted("the entity mapping on disk");
    }
  }

  // 3. The shared metric (over the storage scalar) and the f32 casts. The
  //    SCAN layer of a `binary` index is compared by HAMMING distance over the
  //    sign bits (`toUsearchScanMetric`), not the index's cosine metric --
  //    the fine rerank layer below always carries the index metric.
  impl.scan_.metric_.emplace(impl.meta_.config_.dimensions_,
                             toUsearchScanMetric(impl.meta_.config_.scalar_,
                                                 impl.meta_.config_.metric_),
                             toUsearchScalar(impl.meta_.config_.scalar_));
  impl.scan_.casts_ =
      uu::casts_punned_t::make(toUsearchScalar(impl.meta_.config_.scalar_));

  // 3b. The optional fine RERANK layer of a two-layer index: the same rows in
  //     the same order at the rerank precision, always with the natural
  //     (unpadded) stride. Same metric kind, its own scalar/casts. (Reset
  //     first so re-opening a single-layer index over a previously two-layer
  //     one never keeps a stale layer.)
  impl.rerank_.reset();
  if (impl.meta_.config_.rerankScalar_.has_value()) {
    const VectorScalar rerankScalar = impl.meta_.config_.rerankScalar_.value();
    auto& rerank = impl.rerank_.emplace();
    openMmap(rerank.data_, vectorRerankDataFile(basename, name),
             ad_utility::AccessPattern::Random);
    rerank.dim_ = impl.dim();
    rerank.rowBytes_ = rowBytesFor(rerankScalar, impl.dim());
    rerank.stride_ = rerank.rowBytes_;
    if (rerank.data_.size() != impl.meta_.numVectors_ * rerank.stride_) {
      complainInterrupted("the rerank data size on disk");
    }
    rerank.metric_.emplace(impl.meta_.config_.dimensions_,
                           toUsearchMetric(impl.meta_.config_.metric_),
                           toUsearchScalar(rerankScalar));
    rerank.casts_ = uu::casts_punned_t::make(toUsearchScalar(rerankScalar));
  }

  // 3c. The optional csls hubness sidecar `.csls`: one f32 r(d) per row,
  //     aligned to the flat store. (Reset first so re-opening a non-csls
  //     index over a csls one never keeps a stale mapping.)
  impl.cslsR_ = ad_utility::MmapVectorView<float>{};
  if (impl.meta_.config_.csls_) {
    openMmap(impl.cslsR_, vectorCslsFile(basename, name),
             ad_utility::AccessPattern::Random);
    if (impl.cslsR_.size() != impl.meta_.numVectors_) {
      complainInterrupted("the csls sidecar size on disk");
    }
    // The on-disk r(d) distribution the RUNTIME will actually use, computed
    // fresh in THIS (server) process and independent of any build-time log --
    // same min/p50/p95/max shape so it can be compared directly, at full float
    // precision (setprecision(9)) so a near-1 value cannot masquerade as 1.
    if (impl.cslsR_.size() > 0) {
      const CslsRdSummary s = summarizeCslsRd(
          ql::span<const float>(impl.cslsR_.data(), impl.cslsR_.size()));
      AD_LOG_INFO << std::setprecision(9) << "Vector index \"" << name
                  << "\": loaded csls r(d) sidecar: min/p50/p95/max = "
                  << s.min_ << "/" << s.p50_ << "/" << s.p95_ << "/" << s.max_
                  << std::endl;
    }
    if (impl.meta_.calibratedSoftmaxT_.has_value()) {
      AD_LOG_INFO << std::setprecision(9) << "Vector index \"" << name
                  << "\": softmax autoCut T default (build-calibrated) = "
                  << impl.meta_.calibratedSoftmaxT_.value() << std::endl;
    }
  }

  // 4. The optional HNSW graph, memory-mapped read-only via usearch `view`.
  //    The graph is keyed by row; vectors come from the flat store above.
  if (impl.meta_.hasHnsw_) {
    const std::string hnswPath = vectorHnswFile(basename, name);
    impl.graph_.emplace(
        uu::index_config_t{impl.meta_.config_.hnswConnectivity_,
                           impl.meta_.config_.hnswConnectivity_ * 2});
    auto result = impl.graph_->view(uu::memory_mapped_file_t{hnswPath.c_str()});
    if (!result) {
      AD_THROW("Could not memory-map the HNSW file for vector index \"" + name +
               "\": " + result.error.what());
    }
    if (impl.graph_->size() != impl.meta_.numVectors_) {
      complainInterrupted("the HNSW graph on disk");
    }
    // Validate the graph member keys: they are row indices and are used
    // (unchecked, inside usearch's traversal) to read the flat store, so a
    // corrupt `.hnsw` with an out-of-range key must be rejected here rather
    // than cause an out-of-bounds read at query time. O(numVectors), once.
    for (auto it = impl.graph_->cbegin(); it != impl.graph_->cend(); ++it) {
      if (static_cast<uint64_t>(uu::get_key(*it)) >= impl.meta_.numVectors_) {
        complainInterrupted("the HNSW graph on disk");
      }
    }
    // Reserve one search context per pooled slot. QLever's query-thread count
    // (`-j`) may exceed the hardware concurrency; searches beyond the pool
    // size wait briefly for a free slot instead of failing.
    size_t numSlots = std::max<size_t>(
        2 * std::max(1u, std::thread::hardware_concurrency()), 16);
    uu::index_limits_t limits;
    limits.members = impl.graph_->size();
    limits.threads_add = 0;
    limits.threads_search = numSlots;
    if (!impl.graph_->try_reserve(limits)) {
      AD_THROW("Could not allocate the search contexts for vector index \"" +
               name + "\".");
    }
    impl.searchSlots_ = std::make_unique<SearchSlotPool>(numSlots);
  }

  // 5. Optionally make the flat store(s) resident / SIMD-aligned in RAM.
  //    Purely a paging/throughput optimisation applied after a successful
  //    open. The `residency`/`rerankResidency` arguments (the "preload" /
  //    "preloadRerank" serving settings the load hook threads in from
  //    `QLEVER_VECTOR_SEARCH_ENDPOINTS`, default `None` = mmap-only) are
  //    authoritative and applied PER LAYER -- residency is never persisted,
  //    so e.g. the small i8 scan matrix can be mlocked while the bf16 rerank
  //    matrix stays demand-paged.
  makeResident(residency);
  if (impl.rerank_.has_value()) {
    makeLayerResident(impl.rerank_.value(), rerankResidency,
                      impl.meta_.numVectors_, impl.meta_.config_.name_,
                      "rerank");
  }

  // 6. Resolve the NumKong fast paths (P2 batched bf16 cosine + its norm
  //    sidecar, P3 raw Hamming kernel). Done AFTER residency so the sidecar is
  //    built over the FINAL layer base/stride (the aligned RAM copy when
  //    present). The fine (rerank) layer carries the index metric; the scan
  //    layer of a two-layer binary index is Hamming (`toUsearchScanMetric`).
  int fastPathThreads = 1;
#ifdef _OPENMP
  fastPathThreads = std::max(1, omp_get_max_threads());
#endif
  // The scan layer carries the index metric UNLESS it is a binary index,
  // whose scan layer is Hamming; but the P2 bf16-cosine branch only fires for
  // a bf16 scalar (never binary), so passing the index metric is safe for the
  // metric check and the P3 branch keys on the scalar alone.
  configureLayerFastPaths(impl.scan_, impl.meta_.config_.scalar_,
                          impl.meta_.config_.metric_, impl.meta_.numVectors_,
                          fastPathThreads);
  if (impl.rerank_.has_value()) {
    configureLayerFastPaths(impl.rerank_.value(),
                            impl.meta_.config_.rerankScalar_.value(),
                            impl.meta_.config_.metric_, impl.meta_.numVectors_,
                            fastPathThreads);
  }
}

// ____________________________________________________________________________
// Apply a RAM-residency strategy to one matrix layer (`layerLabel` only names
// the layer in the log lines). See `VectorIndex::makeResident`.
template <typename LayerT>
void makeLayerResident(LayerT& layer, VectorIndex::Residency residency,
                       size_t numRows, const std::string& indexName,
                       const char* layerLabel) {
  using Residency = VectorIndex::Residency;
  const size_t matrixBytes = layer.data_.size();
  if (residency == Residency::None || matrixBytes == 0) {
    return;
  }
  // Fits-in-RAM gate: never drive the machine into OOM/thrash. If we cannot
  // determine the physical memory, be conservative and skip. `AlignedCopy`
  // additionally holds a SECOND full copy, so require it to fit within ~45% of
  // RAM (matrix + its aligned copy <= ~90% of RAM); the others only prefault /
  // pin the single mapped copy, gated at ~90%. Gated per layer with that
  // layer's byte size.
  const uint64_t ram = totalPhysicalMemoryBytes();
  const uint64_t budget =
      residency == Residency::AlignedCopy ? (ram / 100) * 45 : (ram / 100) * 90;
  if (ram == 0 || matrixBytes > budget) {
    AD_LOG_WARN << "Vector index \"" << indexName << "\": the "
                << (matrixBytes >> 20) << " MiB " << layerLabel
                << " store does not comfortably fit in physical "
                   "memory; skipping the requested RAM preload."
                << std::endl;
    return;
  }
  layer.residency_ = residency;

  switch (residency) {
    case Residency::None:
      break;
    case Residency::Advise:
      layer.data_.prefault();
      break;
    case Residency::Lock:
      layer.data_.prefault();
      if (!layer.data_.lockInMemory()) {
        AD_LOG_WARN << "Vector index \"" << indexName << "\": could not mlock "
                    << "the " << layerLabel
                    << " store (RLIMIT_MEMLOCK?); "
                       "continuing with an unlocked mapping."
                    << std::endl;
      }
      break;
    case Residency::AlignedCopy: {
      // Copy the matrix into a 2 MiB-aligned RAM buffer with the NATURAL
      // (unpadded) row stride. Rows are deliberately NOT padded: the NumKong
      // kernels use unaligned loads, while padding inflates the bytes streamed
      // by the memory-bandwidth-bound whole-index sweep (e.g. +33% for a binary
      // 1152-dim layer, 144 -> 192 B/row -- measurably slower than even the
      // unpadded mmap). The anonymous buffer also gets `MADV_HUGEPAGE`; the
      // 2 MiB start alignment (`HUGEPAGE_ALIGNMENT`) is what lets THP actually
      // back it at fault time (a 64-byte start left `AnonHugePages` at 0 in
      // containers where khugepaged never collapsed the interior), cutting the
      // 4 KiB-page TLB-miss cost that throttles the sweep.
      const size_t rowBytes = layer.rowBytes_;
      const size_t n = numRows;
      void* buf = nullptr;
      if (posix_memalign(&buf, HUGEPAGE_ALIGNMENT, n * rowBytes) != 0 ||
          buf == nullptr) {
        AD_LOG_WARN << "Vector index \"" << indexName
                    << "\": could not allocate the aligned RAM copy of the "
                    << layerLabel << " store; falling back to a prefault."
                    << std::endl;
        layer.data_.prefault();
        return;
      }
      std::unique_ptr<char, AlignedFree> owned{static_cast<char*>(buf)};
#if defined(MADV_HUGEPAGE)
      // Advise BEFORE the first touch, so the copy below faults the buffer
      // in as transparent hugepages directly.
      madvise(owned.get(), n * rowBytes, MADV_HUGEPAGE);
#endif
      if (layer.stride_ == rowBytes) {
        // The store already has the natural stride (every v5 index): one
        // dense copy.
        std::memcpy(owned.get(), layer.base(), n * rowBytes);
      } else {
        // Legacy padded stride: compact row by row.
        for (size_t i = 0; i < n; ++i) {
          std::memcpy(owned.get() + i * rowBytes, layer.rowPtr(i), rowBytes);
        }
      }
      // Repoint the read path at the aligned copy (rowPtr()/graphMetric() read
      // `base()` + `stride_`).
      layer.alignedBuf_ = std::move(owned);
      layer.stride_ = rowBytes;
      break;
    }
  }
}

// ____________________________________________________________________________
void VectorIndex::makeResident(Residency residency) {
  makeLayerResident(impl_->scan_, residency, impl_->meta_.numVectors_,
                    impl_->meta_.config_.name_, "flat");
}

// ____________________________________________________________________________
VectorIndex::Residency VectorIndex::residency() const {
  return impl_->scan_.residency_;
}

// ____________________________________________________________________________
VectorIndex::Residency VectorIndex::rerankResidency() const {
  return impl_->rerank_.has_value() ? impl_->rerank_->residency_
                                    : Residency::None;
}

// ____________________________________________________________________________
const VectorIndexMetadata& VectorIndex::metadata() const {
  return impl_->meta_;
}
// ____________________________________________________________________________
void VectorIndex::setEmbeddingEndpoint(std::optional<std::string> url,
                                       std::optional<std::string> model) {
  if (url.has_value()) {
    impl_->meta_.config_.embeddingUrl_ = std::move(url.value());
  }
  if (model.has_value()) {
    impl_->meta_.config_.embeddingModel_ = std::move(model.value());
  }
}
size_t VectorIndex::dimensions() const { return impl_->dim(); }
size_t VectorIndex::numVectors() const { return impl_->meta_.numVectors_; }
size_t VectorIndex::numLiveVectors() const { return impl_->numLive(); }
VectorMetric VectorIndex::metric() const {
  return impl_->meta_.config_.metric_;
}
bool VectorIndex::hasHnsw() const { return impl_->meta_.hasHnsw_; }
bool VectorIndex::hasRerankLayer() const { return impl_->rerank_.has_value(); }

// ____________________________________________________________________________
bool VectorIndex::hasCsls() const { return impl_->meta_.config_.csls_; }
size_t VectorIndex::cslsNeighbors() const {
  return impl_->meta_.config_.cslsNeighbors_;
}
float VectorIndex::cslsRForRow(size_t row) const {
  AD_CONTRACT_CHECK(hasCsls() && row < impl_->meta_.numVectors_);
  return impl_->cslsR_[row];
}
std::optional<float> VectorIndex::cslsRForEntity(Id entity) const {
  AD_CONTRACT_CHECK(hasCsls());
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  return impl_->cslsR_[row.value()];
}
size_t VectorIndex::cslsRerankFloor() const { return impl_->cslsRerankFloor_; }
void VectorIndex::setCslsRerankFloor(size_t floor) {
  // A floor of 0 would make the rerank loop go nowhere; clamp to 1 (the env
  // parser already rejects 0, this is defense in depth for direct callers).
  impl_->cslsRerankFloor_ = std::max<size_t>(floor, 1);
}

// ____________________________________________________________________________
// The per-index serving defaults of the dynamic `vec:autoCut` cuts. The
// setters clamp/ignore invalid values defensively (the env parser already
// validates them); `nullopt` always resets to "use the constant default".
std::optional<float> VectorIndex::cslsFloorDefault() const {
  return impl_->cslsFloorDefault_;
}
void VectorIndex::setCslsFloorDefault(std::optional<float> floor) {
  if (floor.has_value() && !std::isfinite(floor.value())) {
    return;  // A non-finite floor would poison every CSLS comparison.
  }
  impl_->cslsFloorDefault_ = floor;
}
std::optional<float> VectorIndex::softmaxTemperatureDefault() const {
  return impl_->softmaxTemperatureDefault_;
}
void VectorIndex::setSoftmaxTemperatureDefault(
    std::optional<float> temperature) {
  if (temperature.has_value() &&
      !(std::isfinite(temperature.value()) && temperature.value() > 0.f)) {
    return;  // T <= 0 (or NaN/inf) breaks the softmax.
  }
  impl_->softmaxTemperatureDefault_ = temperature;
}
std::optional<size_t> VectorIndex::softmaxNDefault() const {
  return impl_->softmaxNDefault_;
}
void VectorIndex::setSoftmaxNDefault(std::optional<size_t> n) {
  if (n.has_value()) {
    n = std::max<size_t>(n.value(), 1);  // An empty softmax is meaningless.
  }
  impl_->softmaxNDefault_ = n;
}
std::optional<float> VectorIndex::breadthDefault() const {
  return impl_->breadthDefault_;
}
void VectorIndex::setBreadthDefault(std::optional<float> breadth) {
  if (breadth.has_value()) {
    if (!std::isfinite(breadth.value())) {
      return;
    }
    breadth = std::clamp(breadth.value(), 0.f, 1.f);
  }
  impl_->breadthDefault_ = breadth;
}
std::optional<float> VectorIndex::zcutFractionDefault(int mode) const {
  return (mode >= 0 && mode < 3) ? impl_->zcutFractionDefault_[mode]
                                 : std::nullopt;
}
void VectorIndex::setZcutFractionDefault(int mode,
                                         std::optional<float> fraction) {
  if (mode < 0 || mode >= 3) {
    return;
  }
  if (fraction.has_value() &&
      !(std::isfinite(fraction.value()) && fraction.value() > 0.f &&
        fraction.value() <= 1.f)) {
    return;  // The band fraction must be in (0, 1].
  }
  impl_->zcutFractionDefault_[mode] = fraction;
}
std::optional<float> VectorIndex::zcutGateZDefault() const {
  return impl_->zcutGateZDefault_;
}
void VectorIndex::setZcutGateZDefault(std::optional<float> gateZ) {
  if (gateZ.has_value() &&
      !(std::isfinite(gateZ.value()) && gateZ.value() >= 0.f)) {
    return;
  }
  impl_->zcutGateZDefault_ = gateZ;
}
std::optional<float> VectorIndex::zcutFloorFractionDefault() const {
  return impl_->zcutFloorFractionDefault_;
}
void VectorIndex::setZcutFloorFractionDefault(std::optional<float> fraction) {
  if (fraction.has_value()) {
    if (!std::isfinite(fraction.value()) || fraction.value() <= 0.f ||
        fraction.value() > 1.f) {
      return;  // The background fraction must be in (0, 1].
    }
  }
  impl_->zcutFloorFractionDefault_ = fraction;
}
std::optional<float> VectorIndex::calibratedSoftmaxTemperature() const {
  return impl_->meta_.calibratedSoftmaxT_;
}

// ____________________________________________________________________________
bool VectorIndex::hasVector(Id entity) const {
  return impl_->rowOf(entity).has_value();
}

// ____________________________________________________________________________
void VectorIndex::memberEntities(ql::span<Id> out) const {
  const auto& rowmap = impl_->rowmap_;
  AD_CONTRACT_CHECK(out.size() == rowmap.size());
  // The rowmap is stored strictly ascending by `idBits_` (validated at open),
  // so this single pass emits the members in ascending `ValueId` order.
  for (size_t i = 0; i < rowmap.size(); ++i) {
    out[i] = Id::fromBits(rowmap[i].idBits_);
  }
}

// ____________________________________________________________________________
std::optional<std::vector<float>> VectorIndex::getVector(Id entity) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  // Decode from the FINE layer (the rerank matrix when present): it is the
  // higher-precision representation of the same input vector.
  const auto& layer = impl.fine();
  std::vector<float> out(impl.dim());
  auto toF32 = layer.casts_.to.f32;
  if (toF32 == nullptr || !toF32(layer.rowPtr(row.value()), impl.dim(),
                                 reinterpret_cast<char*>(out.data()))) {
    // Same representation (f32): copy the raw bytes.
    std::memcpy(out.data(), layer.rowPtr(row.value()), layer.rowBytes_);
  }
  return out;
}

// ____________________________________________________________________________
Id distanceToValueId(float distance) {
  return std::isnan(distance)
             ? Id::makeUndefined()
             : Id::makeFromDouble(static_cast<double>(distance));
}

// ____________________________________________________________________________
// The `DistanceComputer` (and the one-shot `distance` overloads below) always
// read the FINE layer -- `impl_->fine()` is the rerank matrix when present --
// so `vec:distance` is the exact baseline of a two-layer index.
float VectorIndex::DistanceComputer::operator()(Id entity) const {
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return impl_->fine().distanceToRow(queryBytes_.data(), row.value());
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::atRow(size_t row) const {
  return impl_->fine().distanceToRow(queryBytes_.data(), row);
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::operator()(
    ql::span<const float> vector) const {
  if (vector.size() != impl_->dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = impl_->fine().encodeQuery(vector, buffer);
  return impl_->fine().distanceBetweenBytes(queryBytes_.data(), encoded);
}

// ____________________________________________________________________________
float VectorIndex::distance(Id a, Id b) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto rowA = impl.rowOf(a);
  auto rowB = impl.rowOf(b);
  if (!rowA.has_value() || !rowB.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return layer.distanceBetweenBytes(layer.rowPtr(rowA.value()),
                                    layer.rowPtr(rowB.value()));
}

// ____________________________________________________________________________
float VectorIndex::distance(Id entity, ql::span<const float> vector) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value() || vector.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = layer.encodeQuery(vector, buffer);
  return layer.distanceToRow(encoded, row.value());
}

// ____________________________________________________________________________
float VectorIndex::distance(ql::span<const float> a,
                            ql::span<const float> b) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  if (a.size() != impl.dim() || b.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> bufferA;
  std::vector<char> bufferB;
  const char* encodedA = layer.encodeQuery(a, bufferA);
  const char* encodedB = layer.encodeQuery(b, bufferB);
  return layer.distanceBetweenBytes(encodedA, encodedB);
}

// ____________________________________________________________________________
VectorIndex::DistanceComputer VectorIndex::makeDistanceComputer(
    ql::span<const float> query) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  // Encode the query into the FINE layer's storage scalar and OWN the
  // resulting bytes (`encodeQuery` may return a pointer into the raw f32
  // input when no conversion is needed, so always copy `rowBytes_` out of it).
  std::vector<char> buffer;
  const char* encoded = layer.encodeQuery(query, buffer);
  std::vector<char> owned(encoded, encoded + layer.rowBytes_);
  return DistanceComputer{&impl, std::move(owned)};
}

// ____________________________________________________________________________
std::optional<VectorIndex::DistanceComputer>
VectorIndex::makeDistanceComputerByEntity(Id entity) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  const char* stored = layer.rowPtr(row.value());
  std::vector<char> owned(stored, stored + layer.rowBytes_);
  return DistanceComputer{&impl, std::move(owned)};
}

namespace {
// Rows per contiguous chunk of the sorted merge-walk: each chunk pays ONE
// `lower_bound` into the rowmap and then walks a forward cursor, so the whole
// scan does ~one binary search per chunk instead of one per row. Also the
// cancellation-poll granularity. Coarse on purpose (the per-row cost is one
// pointer compare plus at most one SIMD kernel call).
constexpr size_t GATHER_CHUNK = 1024;

// Below this many rows the fork/join overhead of the parallel walk is not
// worth it, so a small block stays serial (mirrors the per-row path's
// threshold in `VectorDistanceExpression.cpp`).
constexpr size_t GATHER_PARALLEL_THRESHOLD = 2048;
}  // namespace

// ____________________________________________________________________________
void VectorIndex::gatherSortedDistances(ql::span<const Id> ascendingEntities,
                                        const DistanceComputer& computer,
                                        absl::FunctionRef<Id(size_t)> onMiss,
                                        absl::FunctionRef<bool()> isCancelled,
                                        int numThreads,
                                        ql::span<Id> out) const {
  const auto& impl = *impl_;
  const size_t n = ascendingEntities.size();
  AD_CORRECTNESS_CHECK(out.size() == n);
  if (n == 0) {
    return;
  }
  // The id-sorted rowmap as a contiguous, random-access span so each chunk can
  // `lower_bound` its first id and then advance a local cursor.
  const IdRowPair* rmBegin = impl.rowmap_.data();
  const IdRowPair* rmEnd = rmBegin + impl.rowmap_.size();

  // Process one contiguous chunk [lo, hi) of the ascending entity column. ONE
  // binary search locates the chunk's first id in the rowmap; a LOCAL cursor
  // `j` then walks forward as the (ascending) ids advance, so the matched row
  // is exactly the one `rowOf` would return (same rowmap, same comparator).
  // `j` is per-chunk scratch and every `out[i]` write is disjoint, so chunks
  // are independent and safe to run concurrently.
  auto processChunk = [&](size_t lo, size_t hi) {
    const IdRowPair* j =
        std::lower_bound(rmBegin, rmEnd, ascendingEntities[lo].getBits(),
                         [](const IdRowPair& pair, uint64_t bits) {
                           return pair.idBits_ < bits;
                         });
    for (size_t i = lo; i < hi; ++i) {
      // Consecutive duplicate ids resolve to the same row/distance -- reuse the
      // previous result instead of recomputing the SIMD distance.
      if (i > lo && ascendingEntities[i] == ascendingEntities[i - 1]) {
        out[i] = out[i - 1];
        continue;
      }
      uint64_t bits = ascendingEntities[i].getBits();
      while (j != rmEnd && j->idBits_ < bits) {
        ++j;
      }
      if (j != rmEnd && j->idBits_ == bits) {
        out[i] =
            distanceToValueId(computer.atRow(static_cast<size_t>(j->row_)));
      } else {
        // Not a live member: defer to the caller's per-row handling (a
        // non-member id, UNDEF, or a per-row float-list literal), keeping the
        // fast path bit-identical to the per-row path for these rows too.
        out[i] = onMiss(i);
      }
    }
  };

  const size_t numChunks = (n + GATHER_CHUNK - 1) / GATHER_CHUNK;
#ifdef _OPENMP
  if (numThreads > 1 && n >= GATHER_PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
    for (size_t c = 0; c < numChunks; ++c) {
      // Non-throwing per-chunk cancellation poll: an exception must never
      // unwind out of the OpenMP region, so a cancelled scan just leaves the
      // remaining chunks' caller-prefilled `out[i]` and the caller raises the
      // real cancellation afterwards.
      if (isCancelled()) {
        continue;
      }
      size_t lo = c * GATHER_CHUNK;
      processChunk(lo, std::min(n, lo + GATHER_CHUNK));
    }
    return;
  }
#else
  (void)numThreads;
#endif
  for (size_t c = 0; c < numChunks; ++c) {
    if (isCancelled()) {
      return;
    }
    size_t lo = c * GATHER_CHUNK;
    processChunk(lo, std::min(n, lo + GATHER_CHUNK));
  }
}

// Declared in `VectorIndex.h`. `/proc/cpuinfo` lists one block per LOGICAL
// cpu, blocks separated by blank lines; hyperthread siblings share their
// `(physical id, core id)` pair, so the number of unique pairs is the
// physical-core count.
unsigned physicalCoreCount() {
  static const unsigned count = [] {
    ad_utility::HashSet<uint64_t> cores;
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
      // The integer after the ':' if `line` starts with `key`, else `nullopt`.
      auto valueForKey = [](const std::string& line,
                            std::string_view key) -> std::optional<int64_t> {
        if (!ql::starts_with(line, key)) {
          return std::nullopt;
        }
        size_t colon = line.find(':', key.size());
        if (colon == std::string::npos) {
          return std::nullopt;
        }
        return std::strtoll(line.c_str() + colon + 1, nullptr, 10);
      };
      int64_t physicalId = 0;  // stays 0 on kernels that omit the field
      std::optional<int64_t> coreId;
      auto flushBlock = [&]() {
        if (coreId.has_value()) {
          cores.insert((static_cast<uint64_t>(physicalId) << 32) |
                       static_cast<uint32_t>(coreId.value()));
        }
        physicalId = 0;
        coreId = std::nullopt;
      };
      std::string line;
      while (std::getline(cpuinfo, line)) {
        if (line.empty()) {
          flushBlock();
        } else if (auto v = valueForKey(line, "physical id")) {
          physicalId = v.value();
        } else if (auto v = valueForKey(line, "core id")) {
          coreId = v;
        }
      }
      flushBlock();
    }
    if (!cores.empty()) {
      return static_cast<unsigned>(cores.size());
    }
    // No `/proc/cpuinfo`, or it has no core-topology fields (some
    // containers/ARM): fall back to the logical cpu count, which may include
    // hyperthreads -- acceptable.
    return std::max(1u, std::thread::hardware_concurrency());
  }();
  return count;
}

// Declared in `VectorIndex.h`.
int vectorSearchThreadCap() {
  // Optional operator override: a positive integer in
  // `QLEVER_VECTOR_SEARCH_THREADS` replaces the physical-core-count default
  // (it is still capped by OpenMP's configured maximum below, so
  // `OMP_NUM_THREADS` keeps the last word). Anything else -- unset, empty,
  // malformed, zero, negative, or out of range -- keeps the default. Memoized:
  // the environment does not change at runtime.
  static const std::optional<int> envThreads = []() -> std::optional<int> {
    const char* value = std::getenv("QLEVER_VECTOR_SEARCH_THREADS");
    if (value == nullptr) {
      return std::nullopt;
    }
    char* end = nullptr;
    errno = 0;
    long parsed = std::strtol(value, &end, 10);
    if (end == value || *end != '\0' || errno == ERANGE || parsed <= 0 ||
        parsed > std::numeric_limits<int>::max()) {
      return std::nullopt;
    }
    return static_cast<int>(parsed);
  }();
  int cap = envThreads.value_or(static_cast<int>(physicalCoreCount()));
#ifdef _OPENMP
  return std::max(1, std::min(omp_get_max_threads(), cap));
#else
  return std::max(1, cap);
#endif
}

// Declared in `VectorIndex.h`.
bool vectorSearchFullPrecision() {
  // Memoized: the environment does not change at runtime. Truthy =
  // `1`/`true`/`on`/`yes` (case-insensitive); anything else (unset, empty,
  // `0`, `false`, ...) leaves the default OFF.
  static const bool full = []() {
    const char* value = std::getenv("QLEVER_VECTOR_SEARCH_FULL_PRECISION");
    if (value == nullptr) {
      return false;
    }
    std::string v{value};
    for (char& c : v) {
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return v == "1" || v == "true" || v == "on" || v == "yes";
  }();
  return full;
}

namespace {
#ifdef _OPENMP
// Parallelize the exact top-k distance sweep once it crosses this many
// candidates. Below it (notably the small two-layer RERANK pass over a few
// hundred pruned candidates) the scan stays serial -- the OpenMP fan-out would
// cost more than it saves. Mirrors the `vec:distance` expression's thresholds.
constexpr size_t VEC_SEARCH_PARALLEL_THRESHOLD = 2048;
constexpr size_t VEC_SEARCH_PARALLEL_CHUNK = 1024;
#endif

// A bounded max-heap that keeps the `k` smallest (best) distances seen.
// Implemented as an explicit vector + `std::push_heap`/`pop_heap` -- exactly
// what `std::priority_queue` does underneath, so `offer` keeps bit-identical
// semantics -- so that `mergeLocals` can fold the per-thread heaps of a
// parallel scan FLAT (concatenate the raw entries + one selection) instead of
// draining them entry-by-entry through O(numThreads * k * log k) heap pops.
// That drain ran SERIALLY after the parallel region and dominated the coarse
// whole-index sweep at high thread counts (~1.5 ms of a ~4 ms scan at 32
// threads with k=500).
class TopK {
 public:
  using Entry = std::pair<float, uint64_t>;
  explicit TopK(size_t k) : k_{k} {}
  void offer(float distance, uint64_t entity) {
    if (heap_.size() < k_) {
      heap_.emplace_back(distance, entity);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (distance < heap_.front().first) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = Entry{distance, entity};
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  // Extract ascending by distance (ties by entity bits -- the same total
  // `(distance, id)` pair order the max-heap itself uses, so this matches the
  // old pop-and-reverse extraction exactly).
  std::vector<ScoredEntity> sorted() {
    std::sort(heap_.begin(), heap_.end());
    std::vector<ScoredEntity> out;
    out.reserve(heap_.size());
    for (const auto& [dist, id] : heap_) {
      out.push_back(ScoredEntity{Id::fromBits(id), dist});
    }
    heap_.clear();
    return out;
  }

  // Fold the per-thread partial top-ks of a parallel scan into this (empty)
  // heap: concatenate the raw entries and keep the `k` smallest by the
  // `(distance, id)` pair order. For distinct distances the result is
  // identical to offering every entry individually (the k smallest
  // distances); ties AT the k-boundary resolve deterministically towards the
  // smaller entity bits, independent of the thread count. O(total) selection
  // instead of the old O(total * log k) serial heap drain.
  void mergeLocals(std::vector<TopK>& locals) {
    for (TopK& local : locals) {
      heap_.insert(heap_.end(), local.heap_.begin(), local.heap_.end());
      local.heap_.clear();
    }
    if (heap_.size() > k_) {
      std::nth_element(heap_.begin(),
                       heap_.begin() + static_cast<std::ptrdiff_t>(k_),
                       heap_.end());
      heap_.resize(k_);
    }
    // Restore the heap invariant so further `offer` calls stay valid.
    std::make_heap(heap_.begin(), heap_.end());
  }

 private:
  size_t k_;
  std::vector<Entry> heap_;
};

// A bounded selection of the `m` smallest `(coarseDistance, scoringIndex)`
// pairs by the FULL pair order (distance, then index) -- the CSLS coarse
// ranking `byCoarse`. Distinct from `TopK`, whose `offer` compares the
// distance ONLY (its callers just need the distance multiset, so a boundary
// tie may keep either entity): the CSLS rerank BATCH boundary must be the
// exact `(distance, index)` order, because which tied candidates land in the
// reranked batch determines r(q) and the survivor set. Heavily-tying integer
// Hamming coarse distances make this distinction load-bearing.
class CoarseSelector {
 public:
  using Entry = std::pair<float, uint64_t>;  // (coarseDistance, scoringIndex)
  explicit CoarseSelector(size_t m) : m_{m} {}
  void offer(float distance, uint64_t index) {
    Entry e{distance, index};
    if (heap_.size() < m_) {
      heap_.push_back(e);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (m_ != 0 && e < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = e;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  // Fold per-thread partial selections into this (empty) one -- the same flat
  // concatenate + `nth_element` fold as `TopK::mergeLocals`, keeping the `m`
  // smallest pairs.
  void mergeLocals(std::vector<CoarseSelector>& locals) {
    for (CoarseSelector& local : locals) {
      heap_.insert(heap_.end(), local.heap_.begin(), local.heap_.end());
      local.heap_.clear();
    }
    if (heap_.size() > m_) {
      std::nth_element(heap_.begin(),
                       heap_.begin() + static_cast<std::ptrdiff_t>(m_),
                       heap_.end());
      heap_.resize(m_);
    }
  }
  // The kept pairs ascending by (distance, index) -- exactly what
  // `nth_element(all, m) + sort(prefix)` over the full array would yield.
  std::vector<Entry> drainSorted() {
    std::sort(heap_.begin(), heap_.end());
    std::vector<Entry> out{std::make_move_iterator(heap_.begin()),
                           std::make_move_iterator(heap_.end())};
    heap_.clear();
    return out;
  }

 private:
  size_t m_;
  std::vector<Entry> heap_;
};

// Score `n` items into `top` (a `k`-bounded heap): `rowAndId(i)` returns the
// `(row, id)` of item `i`, or `nullopt` to skip it. This is the GATHER scan
// for candidate-restricted searches, whose rows are scattered (the whole-index
// sweep has its own pointer-walk below). The distance computation -- the
// mmap-heavy SIMD work -- runs in parallel once `n` crosses
// `VEC_SEARCH_PARALLEL_THRESHOLD`, via per-thread heaps merged at the end,
// so a big candidate scan uses every core while the small rerank pass stays
// serial. For distinct distances the parallel result is identical to the
// serial one (the same `k` smallest distances).
template <typename LayerT, typename Fn>
void scanIntoTopK(TopK& top, [[maybe_unused]] size_t k, size_t n, LayerT& layer,
                  const char* queryBytes, std::optional<float> maxDistance,
                  Fn&& rowAndId, const CheckInterruptCallback& checkInterrupt) {
  // Hoist the `maxDistance` optional out of the per-row hot path: `nullopt`
  // (the common case) becomes `+inf`, so the filter is a single always-true
  // compare instead of a per-row `optional::has_value()` branch. Bit-identical
  // (`dist <= +inf` keeps every row, exactly as skipping the check did).
  const float maxDist =
      maxDistance.value_or(std::numeric_limits<float>::infinity());
  // Software-prefetch the scattered candidate row this loop will need a few
  // iterations ahead (the gather reads coarse-ranked rows out of physical
  // order, so the hardware streamer cannot see them coming). Large fine rows
  // (bf16, 2304 B) get one line + a short lookahead; compact rows more.
  const bool compact = layer.stride_ <= 256;
  const size_t pfAhead = compact ? 32 : 4;
  auto prefetch = [&](size_t i) {
    const size_t j = i + pfAhead;
    if (j >= n) {
      return;
    }
    std::optional<std::pair<size_t, uint64_t>> rj = rowAndId(j);
    if (!rj.has_value()) {
      return;
    }
    const char* pf = layer.rowPtr(rj->first);
    __builtin_prefetch(pf, 0, 3);
    if (compact) {
      __builtin_prefetch(pf + 64, 0, 3);
      __builtin_prefetch(pf + 128, 0, 3);
    }
  };
  auto scoreOne = [&](TopK& dst, size_t i) {
    std::optional<std::pair<size_t, uint64_t>> ri = rowAndId(i);
    if (!ri.has_value()) {
      return;
    }
    float dist = layer.distanceToRow(queryBytes, ri->first);
    if (dist <= maxDist) {
      dst.offer(dist, ri->second);
    }
  };
#ifdef _OPENMP
  const int numThreads = vectorSearchThreadCap();
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::vector<TopK> locals(static_cast<size_t>(numThreads), TopK{k});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      TopK& localTop = locals[static_cast<size_t>(omp_get_thread_num())];
      // schedule(static): each thread owns ONE contiguous ~n/numThreads block
      // and streams it start-to-end -- best prefetch locality for this
      // bandwidth-bound sweep, and no dynamic-scheduling overhead. The
      // iteration->thread assignment does not affect the result (the
      // per-thread heaps are merged below). `VEC_SEARCH_PARALLEL_CHUNK`
      // remains the interrupt-poll stride inside each block.
#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < n; ++i) {
        // An exception must never unwind out of the OpenMP region: poll the
        // (throwing) interrupt once per chunk, latch a flag on cancellation,
        // and re-raise once AFTER the region.
        if (cancelled.load(std::memory_order_relaxed)) {
          continue;
        }
        if (VEC_SEARCH_CHECK_CANCELLATION && checkInterrupt &&
            i % VEC_SEARCH_PARALLEL_CHUNK == 0) {
          try {
            checkInterrupt();
          } catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
          }
        }
        prefetch(i);
        scoreOne(localTop, i);
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise the cancellation outside the parallel
                         // region
    }
    top.mergeLocals(locals);
    return;
  }
#endif
  size_t sinceCheck = 0;
  for (size_t i = 0; i < n; ++i) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    prefetch(i);
    scoreOne(top, i);
  }
}

#if USEARCH_USE_NUMKONG
// P2: rows per AMX batched-cosine GEMM block. Sized so the block's dot output
// (`blk` floats) plus its bf16 rows stay comfortably in L2; 512 rows at
// 1152-dim bf16 is ~1.1 MiB of A per block, and the measured sweet spot on
// Sapphire/Emerald Rapids was 512-2048 (all within ~5% of each other).
constexpr size_t VEC_BF16_BATCH_ROWS = 1024;

// P2: the whole-index bf16 cosine sweep via the AMX batched GEMM. Same static
// per-thread partition, tombstone filter, and per-thread `TopK` merge as the
// per-row `scanWholeIndexIntoTopK`, but distances come a block at a time from
// `cosineBlockBf16`. Only called when `layer.batchedCosineBf16_` is true. The
// query is packed ONCE into `packed` (read-only, shared across threads).
template <typename ImplT, typename LayerT>
void scanWholeIndexBatchedBf16(TopK& top, size_t k, ImplT& impl, LayerT& layer,
                               const char* queryBytes, float maxDist,
                               const CheckInterruptCallback& checkInterrupt,
                               [[maybe_unused]] int numThreads) {
  const size_t n = impl.meta_.numVectors_;
  const std::vector<char> packed = layer.packQueryBf16(queryBytes);
  const void* pq = packed.data();
  const float qInv = LayerT::queryInvNormBf16(queryBytes, layer.dim_);
  // Score a contiguous block [blockStart, blockStart+count) into `localTop`
  // via one GEMM, honouring tombstones and the max-distance filter.
  auto scoreBlock = [&](TopK& localTop, size_t blockStart, size_t count,
                        float* dots, float* dists) {
    layer.cosineBlockBf16(blockStart, count, pq, qInv, dots, dists);
    for (size_t j = 0; j < count; ++j) {
      const uint64_t id = impl.keys_[blockStart + j];
      if (id != TOMBSTONE_KEY && dists[j] <= maxDist) {
        localTop.offer(dists[j], id);
      }
    }
  };
#ifdef _OPENMP
  if (numThreads > 1) {
    std::vector<TopK> locals(static_cast<size_t>(numThreads), TopK{k});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      const size_t team = static_cast<size_t>(omp_get_num_threads());
      TopK& localTop = locals[tid];
      const size_t first = n * tid / team;
      const size_t last = n * (tid + 1) / team;
      std::vector<float> dots(VEC_BF16_BATCH_ROWS);
      std::vector<float> dists(VEC_BF16_BATCH_ROWS);
      for (size_t s = first; s < last; s += VEC_BF16_BATCH_ROWS) {
        if (VEC_SEARCH_CHECK_CANCELLATION) {
          if (cancelled.load(std::memory_order_relaxed)) {
            break;
          }
          if (checkInterrupt) {
            try {
              checkInterrupt();
            } catch (...) {
              cancelled.store(true, std::memory_order_relaxed);
              break;
            }
          }
        }
        const size_t count = std::min(VEC_BF16_BATCH_ROWS, last - s);
        scoreBlock(localTop, s, count, dots.data(), dists.data());
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise outside the parallel region
    }
    top.mergeLocals(locals);
    return;
  }
#endif
  std::vector<float> dots(VEC_BF16_BATCH_ROWS);
  std::vector<float> dists(VEC_BF16_BATCH_ROWS);
  size_t sinceCheck = 0;
  for (size_t s = 0; s < n; s += VEC_BF16_BATCH_ROWS) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    const size_t count = std::min(VEC_BF16_BATCH_ROWS, n - s);
    scoreBlock(top, s, count, dots.data(), dists.data());
  }
}
#endif

// The dedicated WHOLE-INDEX top-k sweep (the filtered search keeps the
// generic `scanIntoTopK` above, whose gather callback it genuinely needs).
// Visiting ALL rows 0..numVectors in physical order lets every worker carry a
// RUNNING row pointer (one `p += stride_` add per row) instead of the generic
// path's `base() + row * stride_` multiply, and read the keys column directly
// instead of a per-row callback. `numThreads > 1` runs MANUAL static
// partitioning: each worker owns one contiguous, evenly-split row range
// processed start-to-end, so each core streams its own region of the store --
// the prefetch-friendliest order for this memory-bandwidth-bound scan. Rows
// go into per-thread heaps merged at the end, so the thread assignment is
// irrelevant to the result: the same distances are offered as in the serial
// walk, and the top-k is bit-identical (for distinct distances).
template <typename ImplT, typename LayerT>
void scanWholeIndexIntoTopK(TopK& top, size_t k, ImplT& impl, LayerT& layer,
                            const char* queryBytes,
                            std::optional<float> maxDistance,
                            const CheckInterruptCallback& checkInterrupt,
                            [[maybe_unused]] int numThreads) {
  const size_t n = impl.meta_.numVectors_;
  // Hoist the `maxDistance` optional out of the per-row hot path: `nullopt`
  // (the whole-index common case) becomes `+inf`, so the filter is a single
  // always-true compare, not a per-row `optional::has_value()` branch.
  // Bit-identical (`dist <= +inf` keeps every row).
  const float maxDist =
      maxDistance.value_or(std::numeric_limits<float>::infinity());
#if USEARCH_USE_NUMKONG
  // P2: batched bf16 cosine whole-index sweep. One AMX GEMM per block of
  // contiguous rows (query x block) instead of one angular kernel per row --
  // keeps memory saturated at the width=1 query shape on AMX. The query is
  // packed ONCE (shared, read-only) before the parallel region.
  if (layer.batchedCosineBf16_) {
    scanWholeIndexBatchedBf16(top, k, impl, layer, queryBytes, maxDist,
                              checkInterrupt, numThreads);
    return;
  }
#endif
#ifdef _OPENMP
  if (numThreads > 1) {
    std::vector<TopK> locals(static_cast<size_t>(numThreads), TopK{k});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      // Partition by the ACTUAL team size (OpenMP may grant fewer threads
      // than requested); `tid < team <= numThreads` indexes `locals` safely.
      const size_t team = static_cast<size_t>(omp_get_num_threads());
      TopK& localTop = locals[tid];
      const size_t first = n * tid / team;
      const size_t last = n * (tid + 1) / team;
      const char* p = layer.base() + first * layer.stride_;
      const size_t stride = layer.stride_;
      // Software prefetch for COMPACT rows: at ~144 B/row the per-row kernel
      // is too short for the hardware streamer to keep enough misses in
      // flight, so prefetch the row ~4 KiB ahead explicitly (3 lines cover
      // one stride advance; prefetch never faults, so running past the
      // partition or mapping end is harmless). Large rows (bf16 fine layer)
      // skip this: explicitly prefetching them ~3 rows ahead was measured
      // SLOWER on a Sapphire Rapids box (50 -> 63 ms -- cache pollution /
      // prefetch throttling), so the HW streamer handles the big rows.
      const bool prefetch = stride <= 256;
      for (size_t row = first; row < last; ++row, p += stride) {
        if (prefetch) {
          const char* f = p + 4096;
          __builtin_prefetch(f, 0, 3);
          __builtin_prefetch(f + 64, 0, 3);
          __builtin_prefetch(f + 128, 0, 3);
        }
        // An exception must never unwind out of the OpenMP region: poll the
        // (throwing) interrupt once per chunk, latch a flag on cancellation
        // (also noticing another worker's latch), and re-raise once AFTER the
        // region.
        if (VEC_SEARCH_CHECK_CANCELLATION &&
            (row - first) % VEC_SEARCH_PARALLEL_CHUNK == 0) {
          if (cancelled.load(std::memory_order_relaxed)) {
            break;
          }
          if (checkInterrupt) {
            try {
              checkInterrupt();
            } catch (...) {
              cancelled.store(true, std::memory_order_relaxed);
              break;
            }
          }
        }
        uint64_t id = impl.keys_[row];
        if (id != TOMBSTONE_KEY) {
          float dist = layer.distanceBetweenBytes(queryBytes, p);
          if (dist <= maxDist) {
            localTop.offer(dist, id);
          }
        }
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise the cancellation outside the parallel
                         // region
    }
    top.mergeLocals(locals);
    return;
  }
#endif
  const char* p = layer.base();
  const size_t stride = layer.stride_;
  const bool prefetch = stride <= 256;  // see the parallel loop above
  size_t sinceCheck = 0;
  for (size_t row = 0; row < n; ++row, p += stride) {
    if (prefetch) {
      const char* f = p + 4096;
      __builtin_prefetch(f, 0, 3);
      __builtin_prefetch(f + 64, 0, 3);
      __builtin_prefetch(f + 128, 0, 3);
    }
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    uint64_t id = impl.keys_[row];
    if (id != TOMBSTONE_KEY) {
      float dist = layer.distanceBetweenBytes(queryBytes, p);
      if (dist <= maxDist) {
        top.offer(dist, id);
      }
    }
  }
}

// True iff EVERY live row's id (the strictly ascending `.rowmap` id column)
// appears in `candidates` -- i.e. the candidate set covers the whole live
// index, so a filtered search equals the whole-index one. This is the
// dominant production shape (a broad metadata pre-filter binds every member),
// and this check replaces, for that shape, the serial candBits copy +
// merge-join + O(live) `matched` materialization of the gather path (which
// cost ~7x the parallel sweep itself at 2M candidates).
//
// Runs as a parallel subset merge: each worker owns one contiguous rowmap
// range, finds its candidate start by binary search, and advances through the
// span linearly. SOUND WITHOUT PRECONDITIONS on `candidates`: "covered" is
// only reported when every rowmap id was literally matched by equality
// against an element of the span, which proves set membership regardless of
// the span's order or duplicates. An unsorted span can only cause a FALSE
// NEGATIVE (the caller then falls back to the sort + merge-join gather, which
// handles it correctly). O(live/T + candidates/T) per worker, no allocation.
template <typename RowmapT>
bool candidatesCoverAllRows(const RowmapT& rowmap,
                            ql::span<const Id> candidates,
                            [[maybe_unused]] int numThreads) {
  const size_t n = rowmap.size();
  const size_t m = candidates.size();
  if (n == 0) {
    return true;  // An empty live set is covered vacuously.
  }
  if (m < n) {
    return false;  // Fewer candidates than live rows can never cover.
  }
  const Id* cand = candidates.data();
  // Check rowmap rows [first, last): binary-search the candidate start, then
  // advance linearly. Bails out early on the first unmatched id.
  auto rangeCovered = [&](size_t first, size_t last) {
    size_t j = std::lower_bound(cand, cand + m, rowmap[first].idBits_,
                                [](Id c, uint64_t id) {
                                  return c.getBits() < id;
                                }) -
               cand;
    for (size_t i = first; i < last; ++i) {
      const uint64_t id = rowmap[i].idBits_;
      while (j < m && cand[j].getBits() < id) {
        ++j;
      }
      if (j >= m || cand[j].getBits() != id) {
        return false;
      }
    }
    return true;
  };
#ifdef _OPENMP
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::atomic<bool> covered{true};
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      const size_t team = static_cast<size_t>(omp_get_num_threads());
      const size_t first = n * tid / team;
      const size_t last = n * (tid + 1) / team;
      if (first < last && covered.load(std::memory_order_relaxed) &&
          !rangeCovered(first, last)) {
        covered.store(false, std::memory_order_relaxed);
      }
    }
    return covered.load(std::memory_order_relaxed);
  }
#endif
  return rangeCovered(0, n);
}
}  // namespace

// RAII: advise the flat store of `layer` for a SEQUENTIAL scan while an
// ordered scan runs, restoring RANDOM (the gather default) on exit. This is a
// purely advisory `madvise` read-ahead hint on the memory-mapped store; it
// never affects results. Callers skip it (pass `active = false`) when the
// store is a resident aligned RAM copy (no paging to advise). We call
// `advise()` (a stateless `madvise`) rather than `setAccessPattern()` on
// purpose: the search methods are logically const but share one `Impl` across
// concurrent query threads, and `setAccessPattern` would write the vector's
// non-atomic `_pattern` member (a data race). Concurrent `advise()` calls
// only issue overlapping advisory hints, which is harmless.
template <typename LayerT>
struct SeqScanHint {
  LayerT& layer_;
  bool active_;
  SeqScanHint(LayerT& layer, bool active) : layer_{layer}, active_{active} {
    if (active_) {
      layer_.data_.adviseAccessPattern(ad_utility::AccessPattern::Sequential);
    }
  }
  ~SeqScanHint() {
    if (active_) {
      layer_.data_.adviseAccessPattern(ad_utility::AccessPattern::Random);
    }
  }
};

// ____________________________________________________________________________
// The scalar-agnostic core of the exact search: `queryBytes` is already in the
// storage representation of `layer` -- the FINE layer for `searchExact*` (the
// exact baseline), the coarse SCAN layer for `searchExactCoarse*` (the
// SERVICE's candidate pass). (A template so that it can take the private
// `VectorIndex::Impl` without naming it.)
// Forward declaration (defined further below, in the SAME anonymous namespace
// -- anonymous namespaces merge across blocks in a TU): the O(n)
// counting/histogram select for an INTEGER coarse metric (the binary layer's
// Hamming distance). `searchExactBytes` reuses it so the plain top-k coarse
// path matches the CSLS path (both avoid the O(n log k) heap for a large k like
// the binary rerankK).
namespace {
template <typename LayerT, typename RowFn>
std::vector<std::pair<float, uint64_t>> coarseSweepSelectHistogram(
    const LayerT& layer, const char* queryBytes, size_t n, const RowFn& rowOf,
    bool contiguous, float* coarseDists, size_t topM, size_t distMax,
    const CheckInterruptCallback& checkInterrupt);
}  // namespace

// `integerDistMax`, when set, means THIS layer's metric returns an integer
// distance in [0, integerDistMax] (the binary layer's Hamming count) -- so the
// top-k selection uses the branchless histogram instead of the bounded heap.
// `nullopt` (a float metric: i8/bf16/f32) keeps the heap.
template <typename ImplT, typename LayerT>
std::vector<ScoredEntity> searchExactBytes(
    ImplT& impl, LayerT& layer, const char* queryBytes, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored = nullptr,
    std::optional<size_t> integerDistMax = std::nullopt,
    std::vector<uint64_t>* outRows = nullptr) {
  // `numScored` (optional out-param) reports how many vectors actually had a
  // distance computed -- the WHOLE live set for a whole-index search, or just
  // the candidates that are members for a restricted one. It is NOT the raw
  // candidate count: a candidate without a stored vector is skipped, not
  // scored.
  auto reportScored = [numScored](size_t n) {
    if (numScored != nullptr) {
      *numScored = n;
    }
  };
  // `outRows` (optional out-param): when non-null, receives the STORE ROW of
  // every returned survivor, in the SAME order as the returned `ScoredEntity`
  // vector -- the coarse-with-rows path (`searchExactCoarseWithRows`) so the
  // fine rerank can score exactly these rows without re-deriving them from ids.
  auto emitRows = [outRows](std::vector<uint64_t>&& rows) {
    if (outRows != nullptr) {
      *outRows = std::move(rows);
    }
  };
  // Fill `outRows` (if requested) from the entities of a heap-sweep result
  // that did not carry rows itself (the non-integer coarse metric): one
  // `rowOf` lookup PER SURVIVOR (k of them, O(k log numVectors)) -- NOT the
  // O(numVectors) whole-`.rowmap` merge-join the caller is avoiding.
  auto emitRowsFromEntities = [&](const std::vector<ScoredEntity>& out) {
    if (outRows == nullptr) {
      return;
    }
    std::vector<uint64_t> rows;
    rows.reserve(out.size());
    for (const ScoredEntity& e : out) {
      rows.push_back(static_cast<uint64_t>(impl.rowOf(e.entity_).value_or(0)));
    }
    emitRows(std::move(rows));
  };
  // Clamp `k` (user-supplied, unbounded) to the live vector count AND a hard
  // maximum: it bounds the `TopK` heap and would otherwise be a remote OOM
  // lever.
  k = std::min({k, impl.numLive(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    reportScored(0);
    emitRows({});
    return {};
  }
  TopK top{k};

  // Integer-metric (binary Hamming) top-k via the O(n) histogram: map its
  // (distance, scoring-index) pairs to `ScoredEntity`. `rowOf(i)` -> store row,
  // `idOf(i)` -> entity id bits. Ascending by (distance, index) already. When
  // `outRows` was requested it is ALSO filled from `rowOf(i)` -- free here,
  // the histogram already resolved the row.
  auto histSelect = [&](size_t n, auto&& rowOf, auto&& idOf,
                        bool contiguous) -> std::vector<ScoredEntity> {
    SeqScanHint hint{layer, contiguous && !layer.alignedBuf_};
    std::unique_ptr<float[]> dists(new float[n]);
    auto sel = coarseSweepSelectHistogram(layer, queryBytes, n, rowOf,
                                          contiguous, dists.get(), k,
                                          integerDistMax.value(), checkInterrupt);
    const float maxDist =
        maxDistance.value_or(std::numeric_limits<float>::infinity());
    std::vector<ScoredEntity> out;
    out.reserve(sel.size());
    std::vector<uint64_t> rows;
    if (outRows != nullptr) {
      rows.reserve(sel.size());
    }
    for (const auto& [dist, i] : sel) {
      if (dist <= maxDist) {
        out.push_back(ScoredEntity{Id::fromBits(idOf(i)), dist});
        if (outRows != nullptr) {
          rows.push_back(static_cast<uint64_t>(rowOf(i)));
        }
      }
    }
    emitRows(std::move(rows));
    return out;
  };
  // The whole-index histogram scans every row 0..numVectors with no per-row
  // tombstone check, so it is only correct on a tombstone-free store; the gather
  // variant runs over `matched` (already tombstone-free) so it has no such gate.
  const bool histWholeIndex =
      integerDistMax.has_value() && impl.meta_.numTombstones_ == 0;

  if (!candidates.has_value()) {
    // Whole-index brute force: rows are visited in order, so it is sequential.
    // The distance sweep runs in parallel (per-thread heaps) above the
    // threshold, on the dedicated pointer-walk scan.
    int numThreads = 1;
#ifdef _OPENMP
    if (impl.meta_.numVectors_ >= VEC_SEARCH_PARALLEL_THRESHOLD) {
      numThreads = vectorSearchThreadCap();
    }
#endif
    AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                << "\": whole-index scan (" << impl.numLive() << " vectors, "
                << numThreads << " threads)" << std::endl;
    reportScored(impl.numLive());
    if (histWholeIndex) {
      return histSelect(
          impl.meta_.numVectors_, [](size_t i) { return i; },
          [&](size_t i) { return impl.keys_[i]; }, /*contiguous=*/true);
    }
    SeqScanHint hint{layer, !layer.alignedBuf_};
    scanWholeIndexIntoTopK(top, k, impl, layer, queryBytes, maxDistance,
                           checkInterrupt, numThreads);
    auto out = top.sorted();
    emitRowsFromEntities(out);
    return out;
  }

  const size_t live = impl.numLive();
  // FAST PATH: a candidate set at least as large as the live set very often
  // covers EVERY live vector (the common shape of a broad metadata
  // pre-filter that binds every index member) -- then the filtered top-k IS
  // the whole-index top-k. Detect exact coverage directly on the raw spans
  // (one allocation-free parallel subset merge, `candidatesCoverAllRows`) and
  // route straight to the dedicated whole-index sweep. This skips the serial
  // candidate marshalling below (candBits copy + sort check + merge-join +
  // O(live) `matched` materialization), which at ~2M covering candidates cost
  // ~7x the parallel sweep itself. A false negative here (e.g. an unsorted
  // candidate span) merely falls through to the gather path, whose own
  // exact-coverage re-check makes this branch a pure fast path.
  {
    int checkThreads = 1;
#ifdef _OPENMP
    if (live >= VEC_SEARCH_PARALLEL_THRESHOLD) {
      checkThreads = vectorSearchThreadCap();
    }
#endif
    if (candidatesCoverAllRows(impl.rowmap_, candidates.value(),
                               checkThreads)) {
      AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                  << "\": filtered scan: " << candidates->size()
                  << " candidates cover all " << live
                  << " live vectors -> whole-index sweep (" << live
                  << " vectors, " << checkThreads << " threads)" << std::endl;
      reportScored(live);
      if (histWholeIndex) {
        return histSelect(
            impl.meta_.numVectors_, [](size_t i) { return i; },
            [&](size_t i) { return impl.keys_[i]; }, /*contiguous=*/true);
      }
      SeqScanHint hint{layer, !layer.alignedBuf_};
      scanWholeIndexIntoTopK(top, k, impl, layer, queryBytes, maxDistance,
                             checkInterrupt, checkThreads);
      auto out = top.sorted();
      emitRowsFromEntities(out);
      return out;
    }
  }

  // Restricted search: merge-join the candidate id set against the id-sorted
  // `.rowmap` to find which candidates are live members (`mergeJoinRowmap`
  // emits each member row at most once). This replaces both the old
  // per-candidate `lower_bound` gather (sparse sets) and the whole-index
  // membership scan (dense sets) with a single O(#candidates + #rows) pass
  // whose row reads are sequential. A SUPERSET candidate set -- every live
  // vector is a candidate but arrived unsorted, so the fast path above did
  // not see it -- is still detected after the merge and routed to the
  // dedicated whole-index sweep.
  // NOTE: an empty candidate set deliberately yields an empty result -- the
  // caller restricted the search space to nothing. The branch taken (whole-
  // index sweep vs scattered gather) is logged once the merge count is known.
  std::vector<uint64_t> candBits;
  candBits.reserve(candidates->size());
  for (Id c : candidates.value()) {
    candBits.push_back(c.getBits());
  }
  // The candidate column from an index scan already arrives ValueId-sorted; if
  // it does not (e.g. a hand-built set, or one deduplicated in hit order), sort
  // a local copy so the merge -- and its sequential `.data` access -- still
  // applies. (This also collapses duplicate candidate ids.)
  if (!std::is_sorted(candBits.begin(), candBits.end())) {
    std::sort(candBits.begin(), candBits.end());
  }
  // The merge emits rows in non-decreasing order on a genuinely id-sorted
  // store. Track it: if it is ever violated (a stale collation), latch the flag
  // so later gathers skip the misleading SEQUENTIAL hint.
  bool seqHint = !layer.alignedBuf_ &&
                 !impl.gatherNonMonotonic_.load(std::memory_order_relaxed);
  // Collect the matched (row, id) pairs first: the merge-join is cheap, while
  // the per-row SIMD distance below is the expensive part we parallelize. They
  // arrive in ascending row order on a monotonic store, so the sequential-scan
  // hint still applies while scoring.
  std::vector<std::pair<size_t, uint64_t>> matched;
  matched.reserve(std::min<size_t>(candBits.size(), impl.numLive()));
  bool monotonic = true;
  {
    SeqScanHint hint{layer, seqHint};
    mergeJoinRowmap(candBits.begin(), candBits.end(), impl.rowmap_.begin(),
                    impl.rowmap_.end(), [&](size_t row, uint64_t id) {
                      if (!matched.empty() && row < matched.back().first) {
                        monotonic = false;
                      }
                      matched.emplace_back(row, id);
                    });
  }
  if (seqHint && !monotonic) {
    impl.gatherNonMonotonic_.store(true, std::memory_order_relaxed);
  }
  // Candidate set covers EVERY live vector => the filtered top-k IS the
  // whole-index top-k. Take the dedicated whole-index sweep (a running
  // `p += stride` pointer walk with direct key reads -- no per-row multiply or
  // gather callback) instead of the scattered gather. Correct ONLY at EXACT
  // coverage: with even one live non-candidate, the sweep would score it too
  // and change the result.
  if (matched.size() == live) {
    int numThreads = 1;
#ifdef _OPENMP
    if (live >= VEC_SEARCH_PARALLEL_THRESHOLD) {
      numThreads = vectorSearchThreadCap();
    }
#endif
    AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                << "\": filtered scan: " << candidates->size()
                << " candidates cover all " << live
                << " live vectors -> whole-index sweep (" << live
                << " vectors, " << numThreads << " threads)" << std::endl;
    reportScored(live);
    if (histWholeIndex) {
      return histSelect(
          impl.meta_.numVectors_, [](size_t i) { return i; },
          [&](size_t i) { return impl.keys_[i]; }, /*contiguous=*/true);
    }
    SeqScanHint hint{layer, !layer.alignedBuf_};
    scanWholeIndexIntoTopK(top, k, impl, layer, queryBytes, maxDistance,
                           checkInterrupt, numThreads);
    auto out = top.sorted();
    emitRowsFromEntities(out);
    return out;
  }
  int gatherThreads = 1;
#ifdef _OPENMP
  if (matched.size() >= VEC_SEARCH_PARALLEL_THRESHOLD) {
    gatherThreads = vectorSearchThreadCap();
  }
#endif
  AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
              << "\": filtered scan: " << candidates->size() << " candidates -> "
              << matched.size() << " of " << live
              << " live members (scattered gather, " << gatherThreads
              << " threads)" << std::endl;
  // The members among the candidates -- the rows that actually got a distance.
  reportScored(matched.size());
  if (integerDistMax.has_value()) {
    return histSelect(
        matched.size(), [&](size_t i) { return matched[i].first; },
        [&](size_t i) { return matched[i].second; }, /*contiguous=*/false);
  }
  {
    SeqScanHint hint{layer, seqHint && monotonic};
    scanIntoTopK(
        top, k, matched.size(), layer, queryBytes, maxDistance,
        [&](size_t i) -> std::optional<std::pair<size_t, uint64_t>> {
          return matched[i];
        },
        checkInterrupt);
  }
  auto out = top.sorted();
  emitRowsFromEntities(out);
  return out;
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExact(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  // The FINE layer is binary only for a single-layer binary index (no rerank);
  // then its Hamming distances are integers -> the histogram select applies.
  const VectorScalar fineScalar =
      impl.meta_.config_.rerankScalar_.value_or(impl.meta_.config_.scalar_);
  const std::optional<size_t> integerDistMax =
      fineScalar == VectorScalar::Binary ? std::optional<size_t>{impl.dim()}
                                         : std::nullopt;
  return searchExactBytes(impl, layer, queryBytes, k, candidates, maxDistance,
                          checkInterrupt, numScored, integerDistMax);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactCoarse(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = impl.scan_.encodeQuery(query, buffer);
  // The coarse SCAN layer's binary Hamming distances are integers -> the
  // O(n) histogram select (matches the CSLS coarse pass; avoids the O(n log k)
  // heap that dominated the plain top-k coarse pass at the binary rerankK=50*k).
  const std::optional<size_t> integerDistMax =
      impl.meta_.config_.scalar_ == VectorScalar::Binary
          ? std::optional<size_t>{impl.dim()}
          : std::nullopt;
  return searchExactBytes(impl, impl.scan_, queryBytes, k, candidates,
                          maxDistance, checkInterrupt, numScored, integerDistMax);
}

// ____________________________________________________________________________
std::vector<ScoredRow> VectorIndex::searchExactCoarseWithRows(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = impl.scan_.encodeQuery(query, buffer);
  const std::optional<size_t> integerDistMax =
      impl.meta_.config_.scalar_ == VectorScalar::Binary
          ? std::optional<size_t>{impl.dim()}
          : std::nullopt;
  // Run the identical coarse pass as `searchExactCoarse`, but capture the
  // survivors' store rows (`outRows`) so the fine rerank need not re-derive
  // them (see `ScoredRow` / `searchExactByRows`).
  std::vector<uint64_t> rows;
  auto scored = searchExactBytes(impl, impl.scan_, queryBytes, k, candidates,
                                 maxDistance, checkInterrupt, numScored,
                                 integerDistMax, &rows);
  // `outRows` is filled in lock-step with `scored`, so the two align 1:1.
  AD_CORRECTNESS_CHECK(rows.size() == scored.size());
  std::vector<ScoredRow> out;
  out.reserve(scored.size());
  for (size_t i = 0; i < scored.size(); ++i) {
    out.push_back(ScoredRow{scored[i].entity_, rows[i], scored[i].distance_});
  }
  return out;
}

namespace {
// Core of `searchExactByRows`: score the `rows` (store-row + id survivors of
// the coarse pass) on `layer` against the encoded `queryBytes`, returning the
// top-`k` ascending by distance. `k` is already clamped by the caller.
template <typename LayerT>
std::vector<ScoredEntity> searchExactByRowsBytes(
    LayerT& layer, const char* queryBytes, size_t k,
    ql::span<const ScoredRow> rows, std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) {
  // The rows already ARE the store rows of the coarse survivors (from
  // `searchExactCoarseWithRows`) -- no candidate-id copy/sort and no
  // O(numVectors) `.rowmap` merge-join. Sort a local copy of the (row, id)
  // pairs by row so the scattered gather reads the fine store in ascending
  // physical order (the sequential-scan hint; a no-op under aligned residency).
  // Sorting a small (<= rerankK) set is negligible next to the join it avoids.
  std::vector<std::pair<size_t, uint64_t>> rowIds;
  rowIds.reserve(rows.size());
  for (const ScoredRow& r : rows) {
    rowIds.emplace_back(static_cast<size_t>(r.row_), r.entity_.getBits());
  }
  std::sort(rowIds.begin(), rowIds.end());
  const bool seqHint = !layer.alignedBuf_;
  TopK top{k};
  {
    SeqScanHint hint{layer, seqHint};
    scanIntoTopK(
        top, k, rowIds.size(), layer, queryBytes, maxDistance,
        [&](size_t i) -> std::optional<std::pair<size_t, uint64_t>> {
          return rowIds[i];
        },
        checkInterrupt);
  }
  return top.sorted();
}
}  // namespace

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactByRows(
    ql::span<const float> query, size_t k, ql::span<const ScoredRow> rows,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  auto& impl = *impl_;
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  k = std::min({k, rows.size(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    return {};
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  return searchExactByRowsBytes(layer, queryBytes, k, rows, maxDistance,
                                checkInterrupt);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactByRowsByEntity(
    Id entity, size_t k, ql::span<const ScoredRow> rows,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  auto& impl = *impl_;
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  k = std::min({k, rows.size(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    return {};
  }
  // The query point is the entity's own FINE-layer bytes (no f32 round trip),
  // exactly as `searchExactByEntity` uses them.
  return searchExactByRowsBytes(layer, layer.rowPtr(row.value()), k, rows,
                                maxDistance, checkInterrupt);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactByEntity(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchExactBytes(impl, layer, layer.rowPtr(row.value()), k, candidates,
                          maxDistance, checkInterrupt, numScored);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactCoarseByEntity(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchExactBytes(impl, impl.scan_, impl.scan_.rowPtr(row.value()), k,
                          candidates, maxDistance, checkInterrupt, numScored);
}

// ____________________________________________________________________________
std::vector<ScoredRow> VectorIndex::searchExactCoarseByEntityWithRows(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  // Same coarse-by-entity pass as `searchExactCoarseByEntity`, capturing the
  // survivors' store rows (`outRows`) for the fine rerank.
  std::vector<uint64_t> rows;
  auto scored = searchExactBytes(
      impl, impl.scan_, impl.scan_.rowPtr(row.value()), k, candidates,
      maxDistance, checkInterrupt, numScored, std::nullopt, &rows);
  AD_CORRECTNESS_CHECK(rows.size() == scored.size());
  std::vector<ScoredRow> out;
  out.reserve(scored.size());
  for (size_t i = 0; i < scored.size(); ++i) {
    out.push_back(ScoredRow{scored[i].entity_, rows[i], scored[i].distance_});
  }
  return out;
}

// ____________________________________________________________________________
// The scalar-agnostic core of the HNSW search (see `searchExactBytes`).
template <typename ImplT>
std::vector<ScoredEntity> searchHnswBytes(
    const ImplT& impl, const char* queryBytesIn, size_t k,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) {
  AD_CONTRACT_CHECK(impl.graph_.has_value(),
                    "searchHnsw called but this index has no HNSW structure.");
  // Clamp `k` (user-supplied, unbounded) to the live vector count AND a hard
  // maximum -- it drives the allocations and the search expansion below.
  k = std::min({k, impl.numLive(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    return {};
  }
  // Fetch `k + numTombstones` results: at most `numTombstones` of the nearest
  // can be tombstoned, so this provably captures the k nearest LIVE results in
  // a SINGLE search -- no unbounded retry loop. Cap the over-fetch (and hence
  // the allocation and the search expansion) at a bounded value: on a heavily
  // tombstoned index this may return fewer than k results, which is exactly
  // the documented "rebuild after many removals" case.
  constexpr size_t MAX_HNSW_FETCH = 1'000'000;
  size_t wanted = std::min<size_t>(
      {impl.meta_.numVectors_, k + impl.meta_.numTombstones_, MAX_HNSW_FETCH});
  if (checkInterrupt) {
    checkInterrupt();
  }
  auto slot = impl.searchSlots_->acquire(checkInterrupt);
  FlatStoreMetric metric = impl.graphMetric();
  const auto* queryBytes = reinterpret_cast<const uu::byte_t*>(queryBytesIn);

  uu::index_search_config_t config;
  config.thread = slot.id();
  config.expansion =
      std::max<size_t>(impl.meta_.config_.hnswExpansionSearch_, wanted);
  auto result = impl.graph_->search(queryBytes, wanted, metric, config);
  if (!result) {
    AD_THROW("HNSW search on vector index \"" + impl.meta_.config_.name_ +
             "\" failed: " + result.error.what());
  }
  std::vector<uint64_t> rows(wanted);
  std::vector<uu::distance_punned_t> dists(wanted);
  size_t count = result.dump_to(rows.data(), dists.data());
  std::vector<ScoredEntity> out;
  out.reserve(k);
  for (size_t i = 0; i < count && out.size() < k; ++i) {
    float dist = static_cast<float>(dists[i]);
    // `maxDistance` legitimately yields fewer than k results (there simply are
    // not k neighbours within the distance) -- that is the correct answer, not
    // a reason to fetch more.
    if (maxDistance.has_value() && dist > maxDistance.value()) {
      continue;
    }
    // Defense in depth: a graph member key is a row index; guard the unchecked
    // `keys_` access against a corrupt `.hnsw`.
    if (rows[i] >= impl.meta_.numVectors_) {
      continue;
    }
    uint64_t id = impl.keys_[rows[i]];
    if (id == TOMBSTONE_KEY) {
      continue;
    }
    out.push_back(ScoredEntity{Id::fromBits(id), dist});
  }
  return out;
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnsw(
    ql::span<const float> query, size_t k, std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  const auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  // The graph reads the SCAN layer, so the query is encoded into ITS scalar.
  std::vector<char> buffer;
  const char* queryBytes = impl.scan_.encodeQuery(query, buffer);
  return searchHnswBytes(impl, queryBytes, k, maxDistance, checkInterrupt);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnswByEntity(
    Id entity, size_t k, std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchHnswBytes(impl, impl.scan_.rowPtr(row.value()), k, maxDistance,
                         checkInterrupt);
}

namespace {
// A raw-vector query counts a candidate as the query's exact self-match (and
// excludes it from the r(q) neighbourhood) iff its cosine distance is below
// this. Identical bytes give a distance of exactly 0; the epsilon only covers
// the sqrt rounding of the norm terms. Distinct real embeddings are never
// this close, so at most the genuine self is excluded. (An entity query point
// excludes its own ROW instead -- exact, no epsilon.)
constexpr float CSLS_SELF_EPSILON = 1e-6f;

// One distance sweep of the CSLS search, shared by its passes (the
// single-layer full fine sweep, the two-layer coarse sweep, and the fine
// rerank batches): `dists[i] = layer.distanceToRow(queryBytes, rowOf(i))` for
// `i` in `[0, n)`. The mmap-heavy SIMD work runs in parallel above the usual
// threshold, with the same interrupt-latching pattern as `scanIntoTopK` (an
// exception must never unwind out of the OpenMP region); below it, a serial
// loop with periodic interrupt polls.
// Software-prefetch the row `rowOf` will need `pfAhead` iterations from now,
// hiding the mmap latency of this bandwidth-bound sweep. Works for a
// contiguous scan (coarse whole-index) AND a scattered one (fine rerank of
// coarse-ranked rows): it prefetches the FUTURE ITERATION's row, not a fixed
// byte offset. Compact rows (<=256 B, e.g. the 144 B binary layer) get three
// lines and a longer lookahead; larger rows (bf16 fine) one line, since one
// row already spans many lines and saturates the streamer.
template <typename LayerT, typename RowFn>
inline void prefetchSweepRow(const LayerT& layer, const RowFn& rowOf, size_t i,
                             size_t n, size_t pfAhead, bool compact) {
  if (i + pfAhead >= n) {
    return;
  }
  const char* pf = layer.rowPtr(rowOf(i + pfAhead));
  __builtin_prefetch(pf, 0, 3);
  if (compact) {
    __builtin_prefetch(pf + 64, 0, 3);
    __builtin_prefetch(pf + 128, 0, 3);
  }
}

// `contiguous`: the scoring indices map to store rows one-to-one in order
// (`rowOf(i) == i`, the whole-index full fine sweep) -- then P2's batched bf16
// cosine block path applies. A scattered sweep (`rowOf` is a gather, e.g. the
// CSLS rerank batch of coarse-ranked rows) keeps the per-row path.
template <typename LayerT, typename RowFn>
void cslsDistanceSweep(const LayerT& layer, const char* queryBytes, size_t n,
                       const RowFn& rowOf, float* dists,
                       const CheckInterruptCallback& checkInterrupt,
                       [[maybe_unused]] bool contiguous = false) {
  // Prefetch tuning (see `prefetchSweepRow`): ~4 KiB lookahead for compact
  // rows, a few rows for large ones.
  const bool compact = layer.stride_ <= 256;
  const size_t pfAhead = compact ? 32 : 4;
#ifdef _OPENMP
  const int numThreads = vectorSearchThreadCap();
#endif
#if USEARCH_USE_NUMKONG
  // P2: batched bf16 cosine, only when the scoring indices ARE store rows in
  // order. Query packed once (read-only, shared). Blocks of contiguous rows go
  // through the AMX GEMM; `dists[s+j]` is the cosine distance of row `s + j`.
  if (contiguous && layer.batchedCosineBf16_) {
    const std::vector<char> packed = layer.packQueryBf16(queryBytes);
    const void* pq = packed.data();
    const float qInv = LayerT::queryInvNormBf16(queryBytes, layer.dim_);
    auto block = [&](size_t s, size_t count, float* scratch) {
      layer.cosineBlockBf16(s, count, pq, qInv, scratch, dists + s);
    };
#ifdef _OPENMP
    if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
      std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
      {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        const size_t team = static_cast<size_t>(omp_get_num_threads());
        const size_t first = n * tid / team;
        const size_t last = n * (tid + 1) / team;
        std::vector<float> scratch(VEC_BF16_BATCH_ROWS);
        for (size_t s = first; s < last; s += VEC_BF16_BATCH_ROWS) {
          if (VEC_SEARCH_CHECK_CANCELLATION) {
            if (cancelled.load(std::memory_order_relaxed)) break;
            if (checkInterrupt) {
              try {
                checkInterrupt();
              } catch (...) {
                cancelled.store(true, std::memory_order_relaxed);
                break;
              }
            }
          }
          block(s, std::min(VEC_BF16_BATCH_ROWS, last - s), scratch.data());
        }
      }
      if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
        checkInterrupt();
      }
      return;
    }
#endif
    std::vector<float> scratch(VEC_BF16_BATCH_ROWS);
    size_t sinceCheck = 0;
    for (size_t s = 0; s < n; s += VEC_BF16_BATCH_ROWS) {
      if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        checkInterrupt();
      }
      block(s, std::min(VEC_BF16_BATCH_ROWS, n - s), scratch.data());
    }
    return;
  }
  // P2: per-row bf16 cosine (SCATTERED gather or non-AMX). Hoist the query's
  // reciprocal norm ONCE (else `distanceToRow` recomputes it O(dim) per row)
  // and use the sidecar per-row helper, which is bit-identical to the batched
  // block for the same pair.
  const bool perRowCosineBf16 = layer.batchedCosineBf16_;
  const float qInvCosine =
      perRowCosineBf16 ? LayerT::queryInvNormBf16(queryBytes, layer.dim_) : 0.f;
  auto distAt = [&](size_t i) -> float {
    if (perRowCosineBf16) {
      return layer.cosineToRowBf16(queryBytes, rowOf(i), qInvCosine);
    }
    return layer.distanceToRow(queryBytes, rowOf(i));
  };
#else
  auto distAt = [&](size_t i) -> float {
    return layer.distanceToRow(queryBytes, rowOf(i));
  };
#endif
#ifdef _OPENMP
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::atomic<bool> cancelled{false};
    // schedule(static): each thread streams one contiguous block, so the
    // coarse whole-index sweep (rows in physical order) reads its own region
    // start-to-end -- the prefetch-friendly order, and no dynamic-dispatch
    // overhead. The result is a per-index `dists[i]` write, independent of the
    // iteration->thread mapping.
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (size_t i = 0; i < n; ++i) {
      // An exception must never unwind out of the OpenMP region: poll the
      // (throwing) interrupt once per chunk, latch, re-raise after.
      if (cancelled.load(std::memory_order_relaxed)) {
        continue;
      }
      if (VEC_SEARCH_CHECK_CANCELLATION && checkInterrupt &&
          i % VEC_SEARCH_PARALLEL_CHUNK == 0) {
        try {
          checkInterrupt();
        } catch (...) {
          cancelled.store(true, std::memory_order_relaxed);
          continue;
        }
      }
      prefetchSweepRow(layer, rowOf, i, n, pfAhead, compact);
      dists[i] = distAt(i);
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise outside the parallel region
    }
    return;
  }
#endif
  size_t sinceCheck = 0;
  for (size_t i = 0; i < n; ++i) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    prefetchSweepRow(layer, rowOf, i, n, pfAhead, compact);
    dists[i] = distAt(i);
  }
}

// The two-layer CSLS coarse pass: in one sweep, select the coarse-best `topM`
// (the first rerank batch) via per-thread `CoarseSelector` heaps -- the coarse
// ranking `byCoarse` exactly, but without an `nth_element`/sort over all `n`
// (whose full-n sort was the dominant SERIAL, non-scaling cost of a whole-
// index CSLS query). Returns the batch as `(coarseDistance, scoringIndex)`
// ascending by (distance, index) -- bit-identical to
// `nth_element(keyed, topM) + sort`. If `coarseDists` is non-null every coarse
// distance is ALSO written there (for the rare rerank widening, which then
// re-ranks the rest); the common no-widen path passes null and never
// materializes the full distance array. `rowOf(i)` maps a scoring index to its
// store row (identity for a whole-index scan, the matched row otherwise). Same
// prefetch / static-partition structure as the whole-index sweep.
template <typename LayerT, typename RowFn>
std::vector<std::pair<float, uint64_t>> coarseSweepSelect(
    const LayerT& layer, const char* queryBytes, size_t n, const RowFn& rowOf,
    float* coarseDists, size_t topM,
    const CheckInterruptCallback& checkInterrupt) {
  const bool compact = layer.stride_ <= 256;
  const size_t pfAhead = compact ? 32 : 4;
#ifdef _OPENMP
  const int numThreads = vectorSearchThreadCap();
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::vector<CoarseSelector> locals(static_cast<size_t>(numThreads),
                                       CoarseSelector{topM});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      const size_t team = static_cast<size_t>(omp_get_num_threads());
      CoarseSelector& localSel = locals[tid];
      const size_t first = n * tid / team;
      const size_t last = n * (tid + 1) / team;
      for (size_t i = first; i < last; ++i) {
        if (VEC_SEARCH_CHECK_CANCELLATION &&
            (i - first) % VEC_SEARCH_PARALLEL_CHUNK == 0) {
          if (cancelled.load(std::memory_order_relaxed)) {
            break;
          }
          if (checkInterrupt) {
            try {
              checkInterrupt();
            } catch (...) {
              cancelled.store(true, std::memory_order_relaxed);
              break;
            }
          }
        }
        prefetchSweepRow(layer, rowOf, i, n, pfAhead, compact);
        const float d = layer.distanceToRow(queryBytes, rowOf(i));
        if (coarseDists != nullptr) {
          coarseDists[i] = d;
        }
        localSel.offer(d, static_cast<uint64_t>(i));
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise outside the parallel region
    }
    CoarseSelector merged{topM};
    merged.mergeLocals(locals);
    return merged.drainSorted();
  }
#endif
  CoarseSelector sel{topM};
  size_t sinceCheck = 0;
  for (size_t i = 0; i < n; ++i) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    prefetchSweepRow(layer, rowOf, i, n, pfAhead, compact);
    const float d = layer.distanceToRow(queryBytes, rowOf(i));
    if (coarseDists != nullptr) {
      coarseDists[i] = d;
    }
    sel.offer(d, static_cast<uint64_t>(i));
  }
  return sel.drainSorted();
}

// The two-layer CSLS coarse pass for an INTEGER coarse metric (the binary scan
// layer's Hamming distance is an integer in [0, distMax=dim]): select the
// coarse-best `topM` (the first rerank batch) via a COUNTING/HISTOGRAM select
// instead of a bounded heap. One compute pass writes every Hamming distance
// into `coarseDists[i]` (non-null; reused for the rare widening -- the Hamming
// is NEVER recomputed) and bins it into per-thread histograms over `dim+1`
// small integer buckets (cache-resident, branchless increment). Merging the
// tiny histograms and walking the cumulative counts yields the k-th-smallest
// distance in O(dim); a final O(n) collect pass gathers every row below that
// boundary plus the smallest-scoring-index rows AT the boundary. No `k log k`
// heap, no O(threads*k) merge, so it neither anti-scales with threads nor
// with `k` (unlike `CoarseSelector`, whose ~k*ln(n/k) cache-spilling heap
// pop/pushes dominated a whole-index CSLS query).
//
// BIT-IDENTICAL to `coarseSweepSelect`/`byCoarse`: it returns exactly the
// `topM` smallest `(distance, index)` pairs ascending. Rows below the boundary
// distance are all included; the boundary bucket is filled from the SMALLEST
// scoring indices first (the ascending scan encounters them in index order),
// which is precisely the `(distance, index)` tiebreak, and the final `sort`
// on the raw (exact-integer) float distances reproduces that total order.
template <typename LayerT, typename RowFn>
std::vector<std::pair<float, uint64_t>> coarseSweepSelectHistogram(
    const LayerT& layer, const char* queryBytes, size_t n, const RowFn& rowOf,
    bool contiguous, float* coarseDists, size_t topM, size_t distMax,
    const CheckInterruptCallback& checkInterrupt) {
  const size_t m = std::min(topM, n);
  const size_t nb = distMax + 1;  // buckets 0..distMax
  const bool compact = layer.stride_ <= 256;
  const size_t pfAhead = compact ? 32 : 4;
  // The Hamming distance of `i`, as the EXACT non-negative integer bucket. P1
  // keeps distances as `uint32_t` end to end (no per-row float round-trip):
  // `coarseDists[i]` stores the popcount directly as a float only when a
  // caller needs it (the rare widening); the bucket comes straight from the
  // u32. `rawHamming` is the P3 kernel resolved at open -- the same Hamming
  // usearch computes -- called directly; `nullptr` (non-NumKong build) falls
  // back to the punned metric via `distanceBetweenBytes`.
  auto hammingOf = [&](const char* p) -> uint32_t {
#if USEARCH_USE_NUMKONG
    if (layer.rawHammingKernel_ != nullptr) {
      return layer.rawHamming(queryBytes, p);
    }
#endif
    return static_cast<uint32_t>(layer.distanceBetweenBytes(queryBytes, p));
  };
  auto bucketOfU32 = [nb](uint32_t d) -> size_t {
    return std::min(static_cast<size_t>(d), nb - 1);
  };
  // The compute pass for a scoring-index range: Hamming distance ->
  // `coarseDists[i]` and a histogram bin. `contiguous` (the whole-index /
  // covering scan) carries a RUNNING row pointer -- one `p += stride` add per
  // row and a relative prefetch -- instead of the `rowOf` multiply + double
  // lambda call (a measurable slice at this per-row rate; the coarse compute
  // otherwise matches the plain whole-index sweep). `chunkBreak(i)`, polled
  // once per chunk, returns true to stop (a latched cancellation). P3: the
  // per-row local invariants are hoisted so GCC does not reload them across
  // the opaque kernel call.
  auto computeBins = [&](uint64_t* lh, size_t first, size_t last,
                         const auto& chunkBreak) {
    float* out = coarseDists;
    if (contiguous) {
      const size_t stride = layer.stride_;
      const char* p = layer.base() + first * stride;
      const bool cpt = compact;
      for (size_t i = first; i < last; ++i, p += stride) {
        if (VEC_SEARCH_CHECK_CANCELLATION &&
            (i - first) % VEC_SEARCH_PARALLEL_CHUNK == 0 && chunkBreak(i)) {
          break;
        }
        if (cpt) {
          const char* f = p + 4096;
          __builtin_prefetch(f, 0, 3);
          __builtin_prefetch(f + 64, 0, 3);
          __builtin_prefetch(f + 128, 0, 3);
        }
        const uint32_t d = hammingOf(p);
        out[i] = static_cast<float>(d);
        ++lh[bucketOfU32(d)];
      }
    } else {
      for (size_t i = first; i < last; ++i) {
        if (VEC_SEARCH_CHECK_CANCELLATION &&
            (i - first) % VEC_SEARCH_PARALLEL_CHUNK == 0 && chunkBreak(i)) {
          break;
        }
        prefetchSweepRow(layer, rowOf, i, n, pfAhead, compact);
        const uint32_t d = hammingOf(layer.rowPtr(rowOf(i)));
        out[i] = static_cast<float>(d);
        ++lh[bucketOfU32(d)];
      }
    }
  };
  // Per-thread histograms are RETAINED into the collect phase: the
  // counting-scatter select (P1) derives every thread's exact, non-overlapping
  // write offset into the output from them, so no comparison sort is needed.
  int numThreads = 1;
#ifdef _OPENMP
  numThreads = vectorSearchThreadCap();
  const bool parallel = n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1;
#else
  const bool parallel = false;
#endif
  // `parallel` -> the actual OpenMP team size may be < numThreads; the compute
  // pass records how many threads (`team`) actually partitioned the rows so
  // the collect phase uses the same partition. 1 in the serial case.
  size_t team = 1;
  std::vector<std::vector<uint64_t>> localHists(
      parallel ? static_cast<size_t>(numThreads) : size_t{1},
      std::vector<uint64_t>(nb, 0));
  std::vector<uint64_t> hist(nb, 0);
#ifdef _OPENMP
  if (parallel) {
    std::atomic<bool> cancelled{false};
    std::atomic<size_t> teamSize{0};
    // An exception must never unwind out of the OpenMP region: poll the
    // (throwing) interrupt once per chunk, latch, re-raise after.
    auto chunkBreak = [&](size_t) -> bool {
      if (cancelled.load(std::memory_order_relaxed)) {
        return true;
      }
      if (checkInterrupt) {
        try {
          checkInterrupt();
        } catch (...) {
          cancelled.store(true, std::memory_order_relaxed);
          return true;
        }
      }
      return false;
    };
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      const size_t t = static_cast<size_t>(omp_get_num_threads());
      if (tid == 0) {
        teamSize.store(t, std::memory_order_relaxed);
      }
      const size_t first = n * tid / t;
      const size_t last = n * (tid + 1) / t;
      computeBins(localHists[tid].data(), first, last, chunkBreak);
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise outside the parallel region
    }
    team = teamSize.load(std::memory_order_relaxed);
    for (size_t t = 0; t < team; ++t) {
      for (size_t b = 0; b < nb; ++b) {
        hist[b] += localHists[t][b];
      }
    }
  } else
#endif
  {
    // Serial: the (throwing) interrupt polled directly once per chunk.
    auto chunkBreak = [&](size_t) -> bool {
      if (checkInterrupt) {
        checkInterrupt();
      }
      return false;
    };
    computeBins(localHists[0].data(), 0, n, chunkBreak);
    hist = localHists[0];
    team = 1;
  }
  // 2. Boundary bucket `tb` = smallest distance whose cumulative count reaches
  //    `m`; `below` rows are strictly closer, `need` come from the boundary.
  size_t tb = 0;
  size_t below = 0;
  {
    size_t cum = 0;
    for (size_t b = 0; b < nb; ++b) {
      cum += hist[b];
      if (cum >= m) {
        tb = b;
        below = cum - hist[b];
        break;
      }
    }
  }
  // `need` = boundary-bucket rows kept = m - below (>= 1 whenever m >= 1);
  // used implicitly as the [below, m) tb output slot range below.
  using Pair = std::pair<float, uint64_t>;
  std::vector<Pair> sel(m);
  // 3. Counting-scatter select (P1): place every selected pair at its exact
  //    final sorted position with NO comparison sort and NO growing per-thread
  //    vectors. The sorted order is (distance asc, index asc) -- and distances
  //    ARE the integer buckets, so:
  //      * buckets [0, tb) are fully included; a pair with distance b lands in
  //        the output slot range [base[b], base[b+1]) where base[b] is the
  //        count of all selected pairs in buckets < b;
  //      * bucket tb contributes exactly its `need` SMALLEST-index rows, into
  //        slots [below, below+need) == [below, m).
  //    Within a bucket, output slots must be filled index-ascending. Threads
  //    own ascending, contiguous index ranges and scan them ascending, so a
  //    per-(thread,bucket) prefix over the retained `localHists` gives each
  //    thread the exact starting slot for its rows of each bucket; it then
  //    writes them sequentially. Bit-identical to `nth_element + sort` of the
  //    keyed pairs (which the old path did) because both realise the same
  //    total (distance, index) order.
  //
  // base[b] = number of selected pairs strictly closer than bucket b.
  std::vector<uint64_t> base(nb + 1, 0);
  for (size_t b = 0; b < tb; ++b) {
    base[b + 1] = base[b] + hist[b];
  }
  // Cursors: `cursor[t*nb + b]` = next output slot for thread t's bucket-b
  // rows. For buckets < tb: base[b] + (bucket-b rows in threads < t). For
  // bucket tb: `below` + (tb rows in threads < t), but only the first `need`
  // globally are kept -- a thread writes tb rows only while its running global
  // tb-count is < need.
  const size_t nt = team;
  std::vector<uint64_t> cursor(nt * nb, 0);
  {
    // Prefix over threads for every bucket in [0, tb): start each thread after
    // all lower-thread rows of that bucket.
    for (size_t b = 0; b < tb; ++b) {
      uint64_t run = base[b];
      for (size_t t = 0; t < nt; ++t) {
        cursor[t * nb + b] = run;
        run += localHists[t][b];
      }
    }
    // Bucket tb: prefix the tb output cursor and cap the global count at need.
    uint64_t run = below;  // first tb slot
    for (size_t t = 0; t < nt; ++t) {
      cursor[t * nb + tb] = run;
      run += localHists[t][tb];
    }
  }
  // Scatter chunk `c` (of `nt` fixed chunks, matching the compute partition):
  // its rows go to the exact output slots the cursor prefix reserved for it.
  // Driven by an `omp for` over the FIXED chunk count, so the result is
  // correct regardless of how many threads the runtime actually grants this
  // region (each chunk owns disjoint output slots by construction).
  auto scatterChunk = [&](size_t c) {
    const size_t first = n * c / nt;
    const size_t last = n * (c + 1) / nt;
    uint64_t* cur = cursor.data() + c * nb;
    uint64_t tbCursor = cur[tb];
    const uint64_t tbEnd = m;  // == below + need
    for (size_t i = first; i < last; ++i) {
      const float d = coarseDists[i];
      const size_t b = bucketOfU32(static_cast<uint32_t>(d));
      if (b < tb) {
        sel[cur[b]++] = Pair{d, static_cast<uint64_t>(i)};
      } else if (b == tb && tbCursor < tbEnd) {
        sel[tbCursor++] = Pair{d, static_cast<uint64_t>(i)};
      }
    }
  };
#ifdef _OPENMP
  if (parallel) {
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (size_t c = 0; c < nt; ++c) {
      scatterChunk(c);
    }
  } else
#endif
  {
    scatterChunk(0);
  }
  return sel;
}

// Apply the KNEE autoCut (`CslsCut::Mode::Knee`) to `survivors` -- the
// (maxDistance-filtered) reranked candidates with `csls >= floor`, in any
// order. Sort them by CSLS DESCENDING, inspect only the top `cut.maxKeep_`
// (tail noise never moves the knee), find the largest consecutive gap
// `csls[i] - csls[i+1]`, and cut AFTER its argmax -- but only if that gap is
// SIGNIFICANT, i.e. `> cut.significanceFactor_ x` the (lower) median gap of
// the inspected head. Without a significant knee the cut degrades to the
// fixed floor ("keep everything >= cslsFloor"), which keeps smooth /
// cluster-free score distributions stable instead of cutting at an arbitrary
// jitter maximum. `reranked` (the size of the reranked set the survivors came
// from) is only logged. The caller re-sorts by distance afterwards.
std::vector<CslsScoredEntity> applyCslsKneeCut(
    std::vector<CslsScoredEntity> survivors, const CslsCut& cut,
    size_t reranked, std::string_view indexName) {
  // CSLS descending; ties break by entity id, so the knee rank (and with it
  // the whole result) is deterministic.
  ql::ranges::sort(
      survivors, [](const CslsScoredEntity& a, const CslsScoredEntity& b) {
        return a.csls_ != b.csls_ ? a.csls_ > b.csls_
                                  : a.entity_.getBits() < b.entity_.getBits();
      });
  const size_t head = std::min(survivors.size(), cut.maxKeep_);
  if (head < 2) {
    // Zero or one survivor has no gaps -- nothing to knee, keep as is (the
    // floor-fallback behaviour).
    AD_LOG_INFO << "Vector index \"" << indexName
                << "\": csls knee: " << reranked
                << " reranked -> too few survivors for a knee -> "
                << survivors.size() << " kept (floor " << cut.threshold_ << ")"
                << std::endl;
    return survivors;
  }
  std::vector<float> gaps(head - 1);
  size_t argmax = 0;
  for (size_t i = 0; i + 1 < head; ++i) {
    gaps[i] = survivors[i].csls_ - survivors[i + 1].csls_;
    if (gaps[i] > gaps[argmax]) {
      argmax = i;  // First maximum on ties -- deterministic.
    }
  }
  const float maxGap = gaps[argmax];
  // The region's "typical" gap: the (lower) median of the head's gaps.
  std::vector<float> sortedGaps = gaps;
  auto mid = sortedGaps.begin() +
             static_cast<std::ptrdiff_t>((sortedGaps.size() - 1) / 2);
  std::nth_element(sortedGaps.begin(), mid, sortedGaps.end());
  const float medianGap = *mid;
  if (maxGap > cut.significanceFactor_ * medianGap) {
    survivors.resize(argmax + 1);
    AD_LOG_INFO << "Vector index \"" << indexName
                << "\": csls knee: " << reranked << " reranked -> knee at rank "
                << argmax << ", gap " << maxGap << " -> " << survivors.size()
                << " kept (floor " << cut.threshold_ << ")" << std::endl;
    return survivors;
  }
  AD_LOG_INFO << "Vector index \"" << indexName << "\": csls knee: " << reranked
              << " reranked -> no significant knee (max gap " << maxGap
              << " <= " << cut.significanceFactor_ << " * median gap "
              << medianGap << ") -> " << survivors.size() << " kept (floor "
              << cut.threshold_ << ")" << std::endl;
  return survivors;
}

// Apply the SOFTMAX autoCut (`CslsCut::Mode::Softmax`) over the reranked
// candidates, addressed by `dist(j)` (fine cosine distance) and `idBits(j)`
// for `j` in `[0, count)`: take the top-`cut.softmaxN_` by cosine (capped at
// `count`), softmax their cosines at temperature `cut.temperature_`, and keep
// the standouts `p_i >= cut.alpha_ / N` (N the EFFECTIVE, capped count -- the
// uniform reference of the bar). A near-uniform ("shrugging") distribution
// has every `p_i ~ 1/N < alpha/N`, so NOTHING is kept -- that is the no-match
// rejection. Survivors carry their fine cosine distance as the score and
// `csls_ = NaN` (the mode defines no CSLS value); `maxDistance` filters the
// OUTPUT only. Returned ascending by distance already.
template <typename DistFn, typename IdBitsFn>
std::vector<CslsScoredEntity> applyCslsSoftmaxCut(
    size_t count, const DistFn& dist, const IdBitsFn& idBits,
    const CslsCut& cut, std::optional<float> maxDistance,
    std::string_view indexName) {
  std::vector<CslsScoredEntity> out;
  const size_t n = std::min(cut.softmaxN_, count);
  if (n == 0) {
    return out;
  }
  // The top-n by fine distance ascending; ties break by entity id.
  auto byDistanceThenId = [&](size_t a, size_t b) {
    return dist(a) != dist(b) ? dist(a) < dist(b) : idBits(a) < idBits(b);
  };
  std::vector<size_t> top(count);
  std::iota(top.begin(), top.end(), size_t{0});
  if (n < count) {
    std::nth_element(top.begin(), top.begin() + static_cast<std::ptrdiff_t>(n),
                     top.end(), byDistanceThenId);
    top.resize(n);
  }
  std::sort(top.begin(), top.end(), byDistanceThenId);
  // p_j = softmax(cos_j / T) over the top-n, computed max-shifted (the
  // largest exponent is exactly 0) for numeric stability at small T.
  std::vector<double> p(n);
  const double t = cut.temperature_;
  const double maxCos = 1.0 - static_cast<double>(dist(top[0]));
  double sum = 0;
  for (size_t j = 0; j < n; ++j) {
    p[j] = std::exp(((1.0 - static_cast<double>(dist(top[j]))) - maxCos) / t);
    sum += p[j];
  }
  // The distribution's variance around its (exact) mean 1/n -- the logged
  // confidence signal: ~0 = shrugging, large = a standout exists.
  double variance = 0;
  const double uniform = 1.0 / static_cast<double>(n);
  for (double& pj : p) {
    pj /= sum;
    variance += (pj - uniform) * (pj - uniform);
  }
  variance /= static_cast<double>(n);
  const double bar = static_cast<double>(cut.alpha_) / static_cast<double>(n);
  for (size_t j = 0; j < n; ++j) {
    if (p[j] < bar) {
      // `top` is distance-ascending, so `p` is non-increasing: the first
      // below-bar candidate ends the standouts.
      break;
    }
    const float d = dist(top[j]);
    if (maxDistance.has_value() && d > maxDistance.value()) {
      continue;
    }
    out.push_back(CslsScoredEntity{Id::fromBits(idBits(top[j])), d,
                                   std::numeric_limits<float>::quiet_NaN()});
  }
  AD_LOG_INFO << "Vector index \"" << indexName << "\": csls softmax: top-" << n
              << " variance " << variance << ", kept " << out.size() << " (T "
              << cut.temperature_ << ", alpha " << cut.alpha_ << ")"
              << std::endl;
  return out;
}

// A robust estimate of the background NOISE FLOOR of a reranked window's
// signal scores: the median (mu) and MAD-derived spread (sigma) of its LOW
// `floorFraction` -- the coarse-worst reranked candidates, which are
// increasingly pure background as the rerank widens. `sig` is REORDERED
// in place (partial selections, never a full sort). O(window).
struct NoiseFloor {
  float mu_ = 0.f;
  float sigma_ = 0.f;
};
inline NoiseFloor estimateNoiseFloor(std::vector<float>& sig,
                                     float floorFraction) {
  const size_t m = sig.size();
  if (m == 0) {
    return {};
  }
  // The background is the `bg` SMALLEST signal values (>= 1, <= m).
  size_t bg = static_cast<size_t>(
      std::floor(static_cast<double>(floorFraction) * static_cast<double>(m)));
  bg = std::clamp<size_t>(bg, 1, m);
  // Partition so [0, bg) holds the bg smallest, then take their median.
  if (bg < m) {
    std::nth_element(sig.begin(), sig.begin() + static_cast<std::ptrdiff_t>(bg),
                     sig.end());
  }
  auto median = [](float* first, size_t count) -> float {
    const size_t mid = count / 2;
    std::nth_element(first, first + static_cast<std::ptrdiff_t>(mid),
                     first + static_cast<std::ptrdiff_t>(count));
    return first[mid];
  };
  NoiseFloor f;
  f.mu_ = median(sig.data(), bg);
  // MAD = median absolute deviation from mu; sigma = 1.4826 * MAD (the
  // normal-consistent std estimate). Reuse the bg prefix as scratch.
  for (size_t i = 0; i < bg; ++i) {
    sig[i] = std::abs(sig[i] - f.mu_);
  }
  f.sigma_ = ZCUT_MAD_TO_SIGMA * median(sig.data(), bg);
  return f;
}

// Apply the TOP-ANCHORED Z-CUT to a reranked set: keep the candidates whose
// `signal` lies within a FRACTION `cut.fraction_` of the way down from the BEST
// to the noise-floor median -- `score >= s_max - fraction*(s_max - mu)`, i.e.
// `>= (1-fraction)*s_max + fraction*mu`. Anchoring to the top (not the floor)
// keeps a bounded band below the best even on a SMOOTH cross-modal gradient
// (where a floor-anchored cut over-returns); the fraction dial is query-
// independent (`f=0` -> best only, `f=1` -> down to the floor) unlike a sigma-
// unit width. Scale-INVARIANT (s_max and mu scale together). `exact`
// (keepAtLeastOne_ false) additionally GATES on the best clearing the floor
// (`s_max >= mu + gateZ*sigma`, still sigma-based), returning nothing
// otherwise; the other modes always keep at least the single best. `csls[j]` is
// candidate j's CSLS value for `vec:bindCsls` (NaN for the cosine signal).
inline std::vector<CslsScoredEntity> applyZCut(
    const std::vector<uint64_t>& entityBits, const std::vector<float>& fineDist,
    const std::vector<float>& signal, const std::vector<float>& csls,
    const CslsCut& cut, std::optional<float> maxDistance,
    std::string_view indexName) {
  const size_t m = signal.size();
  std::vector<CslsScoredEntity> out;
  if (m == 0) {
    return out;
  }
  // Candidates by signal DESCENDING (ties by entity id) -- the keep set is a
  // predictable prefix of this order.
  std::vector<uint32_t> order(m);
  std::iota(order.begin(), order.end(), 0u);
  auto bySignalDesc = [&](uint32_t a, uint32_t b) {
    return signal[a] != signal[b] ? signal[a] > signal[b]
                                  : entityBits[a] < entityBits[b];
  };
  std::sort(order.begin(), order.end(), bySignalDesc);
  const float sMax = signal[order[0]];
  // The noise floor from the low tail (a copy, since the estimator reorders).
  std::vector<float> scratch(signal);
  const NoiseFloor floor = estimateNoiseFloor(scratch, cut.floorFraction_);
  const float sigma = floor.sigma_;
  size_t keep = 0;
  const bool gated = !cut.keepAtLeastOne_;  // `exact`
  // The best must clear the floor gate for `exact`; the others always answer.
  // The gate stays SIGMA-based: `exact` answers only if the best clears the
  // floor by `gateZ` spreads.
  const bool topClears = sigma > 0.f && sMax >= floor.mu_ + cut.gateZ_ * sigma;
  // The keep band is a fraction of the way from the best down to the floor
  // median (query-independent; `>= (1-f)*s_max + f*mu`).
  const float band = sMax - cut.fraction_ * (sMax - floor.mu_);
  if (!gated || topClears) {
    keep = 1;  // the best is always kept once we answer at all
    // A degenerate window with no gap (best == floor: a flat / no-standout
    // field) has band == s_max, so only the single best is kept -- never the
    // whole tied field.
    if (sMax > floor.mu_) {
      for (; keep < m; ++keep) {
        // `order` is signal-descending, so the first below-band ends the set.
        if (signal[order[keep]] < band) {
          break;
        }
      }
    }
  }
  out.reserve(keep);
  for (size_t r = 0; r < keep; ++r) {
    const uint32_t j = order[r];
    const float d = fineDist[j];
    if (maxDistance.has_value() && d > maxDistance.value()) {
      continue;
    }
    out.push_back(CslsScoredEntity{Id::fromBits(entityBits[j]), d, csls[j]});
  }
  AD_LOG_INFO << "Vector index \"" << indexName << "\": csls zcut ("
              << (cut.signal_ == CslsCut::Signal::Csls ? "csls" : "cosine")
              << " signal): s_max " << sMax << " floor mu " << floor.mu_
              << " sigma " << sigma << ", fraction " << cut.fraction_
              << " -> band " << band << " -> keep " << keep
              << (gated && !topClears ? " (gated: no match)" : "") << ", kept "
              << out.size() << std::endl;
  return out;
}

// Apply a fully-resolved `cut` to an already-`rerankCsls`'d set, dispatching on
// its mode. The Threshold/Knee/Softmax branches are byte-for-byte the former
// in-line cut (same inputs -> same output); ZCut is the new noise-floor cut.
// Survivors are returned ascending by cosine DISTANCE (ties by entity id).
template <typename ImplT>
std::vector<CslsScoredEntity> applyCslsCutImpl(
    const ImplT& impl, const CslsReranked& r, const CslsCut& cut,
    std::optional<float> maxDistance) {
  const std::string& name = impl.meta_.config_.name_;
  const size_t m = r.rerankDepth();
  auto byDistanceThenId = [](const CslsScoredEntity& a,
                             const CslsScoredEntity& b) {
    return a.distance_ != b.distance_
               ? a.distance_ < b.distance_
               : a.entity_.getBits() < b.entity_.getBits();
  };
  std::vector<CslsScoredEntity> out;
  if (cut.mode_ == CslsCut::Mode::Softmax) {
    // The softmax SELECTION is byte-for-byte the standalone cut; a non-`exact`
    // coverage mode then guarantees at least the single best result (the top
    // cosine), where the bare softmax would keep nothing.
    out = applyCslsSoftmaxCut(
        m, [&](size_t j) { return r.fineDist_[j]; },
        [&](size_t j) { return r.entityBits_[j]; }, cut, maxDistance, name);
    if (out.empty() && cut.keepAtLeastOne_ && m > 0) {
      size_t best = 0;
      for (size_t j = 1; j < m; ++j) {
        if (r.fineDist_[j] < r.fineDist_[best] ||
            (r.fineDist_[j] == r.fineDist_[best] &&
             r.entityBits_[j] < r.entityBits_[best])) {
          best = j;
        }
      }
      const float d = r.fineDist_[best];
      if (!maxDistance.has_value() || d <= maxDistance.value()) {
        out.push_back(
            CslsScoredEntity{Id::fromBits(r.entityBits_[best]), d,
                             std::numeric_limits<float>::quiet_NaN()});
      }
    }
  } else if (cut.mode_ == CslsCut::Mode::ZCut) {
    const bool csSignal = cut.signal_ == CslsCut::Signal::Csls;
    std::vector<float> signal(m);
    std::vector<float> cslsVals(m);
    for (size_t j = 0; j < m; ++j) {
      const float cosSim = 1.f - r.fineDist_[j];
      if (csSignal) {
        const float c = static_cast<float>(
            2.0 * static_cast<double>(cosSim) - static_cast<double>(r.rq_) -
            static_cast<double>(impl.cslsR_[r.storeRows_[j]]));
        signal[j] = c;
        cslsVals[j] = c;
      } else {
        signal[j] = cosSim;
        cslsVals[j] = std::numeric_limits<float>::quiet_NaN();
      }
    }
    out = applyZCut(r.entityBits_, r.fineDist_, signal, cslsVals, cut,
                    maxDistance, name);
  } else {
    // Threshold / Knee: keep `csls >= threshold`, then (Knee) cut at a
    // significant gap.
    for (size_t j = 0; j < m; ++j) {
      const float d = r.fineDist_[j];
      if (maxDistance.has_value() && d > maxDistance.value()) {
        continue;
      }
      const float csls = static_cast<float>(
          2.0 * (1.0 - static_cast<double>(d)) - static_cast<double>(r.rq_) -
          static_cast<double>(impl.cslsR_[r.storeRows_[j]]));
      if (csls >= cut.threshold_) {
        out.push_back(
            CslsScoredEntity{Id::fromBits(r.entityBits_[j]), d, csls});
      }
    }
    if (cut.mode_ == CslsCut::Mode::Knee) {
      out = applyCslsKneeCut(std::move(out), cut, m, name);
    }
  }
  ql::ranges::sort(out, byDistanceThenId);
  return out;
}
}  // namespace

// ____________________________________________________________________________
// STAGE (a) of the CSLS-filtered search: score the candidate set and rerank it
// to the depth the `cut`'s widen policy asks for, returning the MODE-
// INDEPENDENT `CslsReranked` (entity + store row + fine cosine distance in
// coarse-rank order, plus r(q)). A cut of ANY mode is then applied by a
// separate `apply*` step (`searchCslsBytes` below wraps the two), so the
// engine can cache this stage and re-apply the cut for free on a mode switch.
//
// The rerank DEPTH follows `cut.mode_`:
//  * Threshold/Knee: widen while the cut still reaches the coarse boundary
//    (a candidate the fixed/floor cut could keep may lie past it).
//  * Softmax: one batch bounds it (>= `softmaxN`); the coarse layer ranks the
//    very top well.
//  * ZCut: TOP-ANCHORED depth -- widen until the reranked window's MINIMUM
//    cosine falls below the broadest cut's boundary `s_max -
//    cut.widenFraction_ * (s_max - mu)` (so the cached set already holds
//    everything any coverage mode could keep), or `cut.rerankCap_` candidates
//    are reranked (then `plateauFound_ = false`, logged: no silent truncation).
//
// SINGLE-LAYER index: one FULL sweep on the (only) fine layer -- every
// candidate is reranked, so the depth policy is moot. TWO-LAYER index: the
// full sweep runs on the cheap COARSE scan matrix (`coarseQueryBytes` = the
// query in ITS scalar; unused single-layer) and only the coarse-best
// candidates get a FINE distance, widened as above. r(q) (the only two-layer
// approximation) is the mean cosine similarity of the query to its
// top-`neighbors` nearest RERANKED candidates.
//
// `selfRow`, when set, is the query entity's own row, excluded from the r(q)
// neighbourhood by identity; for a raw query vector the single nearest
// candidate is excluded iff its distance is < `CSLS_SELF_EPSILON`.
template <typename ImplT, typename LayerT>
CslsReranked rerankCsls(ImplT& impl, LayerT& layer, const char* queryBytes,
                        const char* coarseQueryBytes,
                        std::optional<size_t> selfRow, const CslsCut& cut,
                        size_t neighbors,
                        std::optional<ql::span<const Id>> candidates,
                        const CheckInterruptCallback& checkInterrupt,
                        bool fullPrecision) {
  // For Threshold this is the tau, for Knee the floor (identical machinery);
  // unused by Softmax.
  const float threshold = cut.threshold_;
  const bool hasCsls = impl.meta_.config_.csls_;
  // 1. The scoring set: a scoring index `i` in [0, n) maps to a store `row`
  //    and its entity `id`. For the common WHOLE-INDEX case with no tombstones
  //    the store is addressed DIRECTLY (row i, id = keys_[i]) and the
  //    ~n*16-byte `(row,id)` list is never materialized -- that per-query
  //    allocation + serial fill was a dominant cost of a whole-index CSLS
  //    query. The candidate path (a sparse merge-join against the id-sorted
  //    rowmap, which dedups and drops non-members) and the rarer tombstoned
  //    whole-index path still materialize `matched`.
  // A candidate set covering every live vector (a broad metadata pre-filter,
  // the common bound-set shape) is, on a tombstone-free store, exactly the
  // whole index -- detected allocation-free (`candidatesCoverAllRows`) and
  // routed through the same direct addressing, skipping the candBits copy +
  // merge-join + `matched` materialization.
  int coverThreads = 1;
#ifdef _OPENMP
  if (impl.meta_.numVectors_ >= VEC_SEARCH_PARALLEL_THRESHOLD) {
    coverThreads = vectorSearchThreadCap();
  }
#endif
  const bool directWhole =
      impl.meta_.numTombstones_ == 0 &&
      (!candidates.has_value() ||
       candidatesCoverAllRows(impl.rowmap_, candidates.value(), coverThreads));
  std::vector<std::pair<size_t, uint64_t>> matched;  // empty iff directWhole
  size_t n;
  if (directWhole) {
    n = impl.meta_.numVectors_;
  } else if (!candidates.has_value()) {
    matched.reserve(impl.numLive());
    for (size_t row = 0; row < impl.meta_.numVectors_; ++row) {
      uint64_t id = impl.keys_[row];
      if (id != TOMBSTONE_KEY) {
        matched.emplace_back(row, id);
      }
    }
    n = matched.size();
  } else {
    std::vector<uint64_t> candBits;
    candBits.reserve(candidates->size());
    for (Id c : candidates.value()) {
      candBits.push_back(c.getBits());
    }
    if (!std::is_sorted(candBits.begin(), candBits.end())) {
      std::sort(candBits.begin(), candBits.end());
    }
    matched.reserve(candBits.size());
    mergeJoinRowmap(candBits.begin(), candBits.end(), impl.rowmap_.begin(),
                    impl.rowmap_.end(), [&](size_t row, uint64_t id) {
                      matched.emplace_back(row, id);
                    });
    n = matched.size();
  }
  if (n == 0) {
    return {};
  }
  // Scoring-set accessors: index -> store row and entity id (see
  // `directWhole`). Identity/`keys_` reads for a whole-index scan, an indirect
  // `matched` lookup otherwise.
  auto rowAt = [&](size_t i) -> size_t {
    return directWhole ? i : matched[i].first;
  };
  auto idAt = [&](size_t i) -> uint64_t {
    return directWhole ? impl.keys_[i] : matched[i].second;
  };
  CslsReranked result;
  result.scored_ = n;
  result.hasCsls_ = hasCsls;

  if (!impl.rerank_.has_value() || fullPrecision) {
    // SINGLE-LAYER (or FULL-PRECISION forced): the coarse and fine layers
    // coincide -- or the caller's `fullPrecision` deliberately bypasses the
    // coarse layer -- so a coarse preselection would buy nothing (or is
    // suppressed): one full sweep of the FINE matrix (`layer` is always the
    // fine layer). Every candidate is reranked, so the depth policy is moot.
    // Forcing this branch on a two-layer index runs autoCut/CSLS directly on
    // the full-precision layer (e.g. bf16) -- the `vec:fullPrecision` mode.
    //
    // 2. The full fine-layer distance sweep (CSLS needs EVERY cosine, so
    //    there is no top-k shortcut). Rows come out of step 1
    //    (near-)ascending, so the sequential read-ahead hint applies.
    std::vector<float> dists(n);
    {
      SeqScanHint hint{layer, !layer.alignedBuf_};
      // `directWhole` -> `rowAt` is identity -> the batched bf16 path (P2).
      cslsDistanceSweep(
          layer, queryBytes, n, [&](size_t i) { return rowAt(i); },
          dists.data(), checkInterrupt, /*contiguous=*/directWhole);
    }
    // 3. r(q): the mean cosine similarity of the query to its top-`neighbors`
    //    nearest SCORED candidates, the exact self-match excluded.
    size_t excluded = n;  // scoring index; `n` = none
    if (selfRow.has_value()) {
      for (size_t i = 0; i < n; ++i) {
        if (rowAt(i) == selfRow.value()) {
          excluded = i;
          break;
        }
      }
    } else {
      size_t best = 0;
      for (size_t i = 1; i < n; ++i) {
        if (dists[i] < dists[best]) {
          best = i;
        }
      }
      if (dists[best] < CSLS_SELF_EPSILON) {
        excluded = best;
      }
    }
    CslsNeighborhood neighborhood{neighbors};
    for (size_t i = 0; i < n; ++i) {
      if (i != excluded) {
        neighborhood.offer(dists[i]);
      }
    }
    result.rq_ = neighborhood.meanCosSim();
    // The reranked set is ALL candidates (scoring-index order).
    result.entityBits_.resize(n);
    result.storeRows_.resize(n);
    result.fineDist_.assign(dists.begin(), dists.end());
    for (size_t i = 0; i < n; ++i) {
      result.entityBits_[i] = idAt(i);
      result.storeRows_[i] = rowAt(i);
    }
    result.plateauFound_ = true;
    return result;
  }

  // TWO-LAYER: coarse scan of everything, fine rerank of a bounded chunk.
  //
  // 2/3. Coarse sweep + first-batch selection: the coarse-best M =
  //    `cslsRerankFloor` -- the first rerank batch -- comes back in `sel`,
  //    ascending by (coarse distance, index). The binary scan layer's Hamming
  //    distance is a small integer, so it uses an O(n) counting/histogram
  //    select (`coarseSweepSelectHistogram`) that scales with both threads and
  //    k; a float coarse distance (i8's quantized cosine) uses a per-thread
  //    bounded-heap select (`coarseSweepSelect`). Both avoid the former
  //    per-query `nth_element`/sort over ALL `n` candidates that dominated a
  //    whole-index CSLS query. Smaller coarse distance = closer.
  const size_t floorM = std::max<size_t>(impl.cslsRerankFloor_, 1);
  // The binary scan layer's coarse distance is an integer Hamming count in
  // [0, dim] -> a counting/histogram select (O(n), no heap, scales with
  // threads AND k). Every other scan scalar's coarse distance is a quantized
  // FLOAT (cosine/l2sq) -> the bounded-heap select.
  const bool integerCoarse = impl.meta_.config_.scalar_ == VectorScalar::Binary;
  // Materialized eagerly by the histogram path (it needs every distance for
  // the collect pass and reuses it for any widening), lazily by the heap
  // path's `ensureKeyed` on the first widen only.
  std::unique_ptr<float[]> coarseDists;
  std::vector<std::pair<float, uint64_t>> sel;
  {
    SeqScanHint hint{impl.scan_, !impl.scan_.alignedBuf_};
    if (integerCoarse) {
      coarseDists.reset(new float[n]);
      sel = coarseSweepSelectHistogram(
          impl.scan_, coarseQueryBytes, n, [&](size_t i) { return rowAt(i); },
          directWhole, coarseDists.get(), floorM, impl.dim(), checkInterrupt);
    } else {
      // Pass null: the common no-widen path never needs the full
      // coarse-distance array, only the coarse-best batch (`sel`).
      sel = coarseSweepSelect(
          impl.scan_, coarseQueryBytes, n, [&](size_t i) { return rowAt(i); },
          nullptr, floorM, checkInterrupt);
    }
  }
  // The FULL coarse ranking is materialized only if the rerank WIDENS past the
  // first batch (a safety net that rarely fires). Until then `orderIdx(j)`
  // reads the coarse-best batch straight from `sel`. On the first widen,
  // `ensureKeyed` recovers every coarse distance (from `coarseDists` if the
  // histogram path already has it, else a cheap re-sweep of the quantized
  // coarse layer), builds the flat `(coarseDistance, index)` ranking, and
  // canonicalizes its first M entries to the SAME order `sel` holds (so the
  // already-computed fine distances stay aligned), leaving the tail as the
  // complement for further batches -- exactly the state the old all-at-once
  // selection produced.
  std::vector<std::pair<float, size_t>> keyed;  // lazily built on first widen
  auto ensureKeyed = [&]() {
    if (!keyed.empty()) {
      return;
    }
    if (!coarseDists) {
      coarseDists.reset(new float[n]);
      SeqScanHint hint{impl.scan_, !impl.scan_.alignedBuf_};
      cslsDistanceSweep(
          impl.scan_, coarseQueryBytes, n, [&](size_t i) { return rowAt(i); },
          coarseDists.get(), checkInterrupt, /*contiguous=*/directWhole);
    }
    keyed.resize(n);
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads( \
        vectorSearchThreadCap()) if (n >= VEC_SEARCH_PARALLEL_THRESHOLD)
#endif
    for (size_t i = 0; i < n; ++i) {
      keyed[i] = {coarseDists[i], i};
    }
    const size_t m = std::min(floorM, n);
    if (m < n) {
      std::nth_element(keyed.begin(),
                       keyed.begin() + static_cast<std::ptrdiff_t>(m),
                       keyed.end());
    }
    std::sort(keyed.begin(), keyed.begin() + static_cast<std::ptrdiff_t>(m));
  };
  // Coarse-ranked scoring index at reranked position `j`: from `sel` (the
  // first batch) until a widen materializes the full `keyed`.
  auto orderIdx = [&](size_t j) -> size_t {
    return keyed.empty() ? static_cast<size_t>(sel[j].second) : keyed[j].second;
  };
  // The fine cosine distance of `matched[orderIdx(j)]`, filled for j <
  // reranked (only that prefix is ever read; uninitialized to skip the
  // per-query zero-fill of the untouched tail).
  std::unique_ptr<float[]> fineDists{new float[n]};
  size_t reranked = 0;
  float rq = 0.f;

  // 5. r(q) over the reranked prefix [0, count): the mean cosine similarity
  //    of the query to its top-`neighbors` nearest RERANKED candidates, the
  //    exact self-match excluded -- the same computation as the single-layer
  //    step 3, over the reranked set instead of all matched. (A genuine
  //    self-match has coarse distance ~0, so it is always in the first
  //    batch.)
  auto computeRq = [&](size_t count) {
    size_t excluded = count;  // position in the coarse-ranked prefix; none = n
    if (selfRow.has_value()) {
      for (size_t j = 0; j < count; ++j) {
        if (rowAt(orderIdx(j)) == selfRow.value()) {
          excluded = j;
          break;
        }
      }
    } else {
      size_t best = 0;
      for (size_t j = 1; j < count; ++j) {
        if (fineDists[j] < fineDists[best]) {
          best = j;
        }
      }
      if (fineDists[best] < CSLS_SELF_EPSILON) {
        excluded = best;
      }
    }
    CslsNeighborhood neighborhood{neighbors};
    for (size_t j = 0; j < count; ++j) {
      if (j != excluded) {
        neighborhood.offer(fineDists[j]);
      }
    }
    return neighborhood.meanCosSim();
  };
  // The CSLS value of reranked candidate `j` under a given r(q) -- the same
  // arithmetic as the single-layer step 4.
  auto cslsOf = [&](size_t j, float rqNow) {
    return static_cast<float>(
        2.0 * (1.0 - static_cast<double>(fineDists[j])) -
        static_cast<double>(rqNow) -
        static_cast<double>(impl.cslsR_[rowAt(orderIdx(j))]));
  };
  // ZCut (top-anchored) widen state: the LOWEST floor median seen so far. A
  // tight top cluster has mu ~ s_max, so a band anchored on the current
  // window's median would stall inside the cluster; anchoring to the MIN mu
  // (the most conservative -- lowest -- floor) keeps the band low, near the
  // true noise floor, so the widen progresses through the cluster until real
  // background is reranked. Plus the rerank cap (0 = the whole scored set).
  const bool plateauMode = cut.mode_ == CslsCut::Mode::ZCut;
  const size_t rerankCap =
      cut.rerankCap_ == 0 ? n : std::min<size_t>(cut.rerankCap_, n);
  float muMin = std::numeric_limits<float>::infinity();
  bool cappedOut = false;

  while (true) {
    const size_t batchStart = reranked;
    const size_t batchEnd = std::min(batchStart + floorM, n);
    // Select the coarse-best batch. The FIRST batch is already in `sel`
    // (ascending), selected during the sweep. A widening batch (rare)
    // materializes the full `keyed` ranking and partitions the next chunk to
    // the front (`nth_element`), then sorts just that chunk -- never a full
    // sort of all `n`. Plain `<` on the pairs is the coarse ranking (distance,
    // then index tiebreak).
    if (batchStart != 0) {
      ensureKeyed();
      if (batchEnd < n) {
        std::nth_element(keyed.begin() + batchStart, keyed.begin() + batchEnd,
                         keyed.end());
      }
      std::sort(keyed.begin() + batchStart, keyed.begin() + batchEnd);
    }
    if (checkInterrupt) {
      checkInterrupt();
    }
    // 4. Rerank the batch on the FINE layer. Its rows are coarse-ranked, i.e.
    //    scattered -- no sequential hint.
    cslsDistanceSweep(
        layer, queryBytes, batchEnd - batchStart,
        [&](size_t j) { return rowAt(orderIdx(batchStart + j)); },
        fineDists.get() + batchStart, checkInterrupt);
    reranked = batchEnd;
    // Widening only ever ADDS candidates to the r(q) neighbourhood pool, so
    // r(q) is non-decreasing across batches and the widen decisions below --
    // taken with the then-current r(q) -- are conservative w.r.t. the final
    // cut (a candidate passing the final cut also passes it here). The
    // Softmax mode never consults r(q).
    if (cut.mode_ != CslsCut::Mode::Softmax) {
      rq = computeRq(reranked);
    }
    if (reranked == n) {
      break;
    }
    // 7. Widen (a safety net that rarely fires)? First batch: only when the
    //    coarse-WORST reranked candidate (the M-th) passes the cut -- then
    //    the cut region may extend past the coarse boundary. Later batches:
    //    while the batch yields ANY survivor. Neither holds => a whole
    //    coarse-contiguous batch was rejected, so the (coarse-worse) rest is
    //    left unranked. `maxDistance` is deliberately ignored here: it only
    //    filters the output, and ignoring it can only widen further.
    //    Softmax mode: it only ever looks at the top-`softmaxN` fine cosines
    //    (the coarse layer ranks the very top well -- the same approximation
    //    r(q) already makes), so ONE batch bounds it; widen only while the
    //    batch is smaller than `softmaxN` (i.e. `cslsRerankFloor` was set
    //    below it) -- never toward a full fine scan.
    bool widen;
    if (plateauMode) {
      // TOP-ANCHORED widen: rerank until the window's MINIMUM cosine falls
      // below the broadest cut's boundary `s_max - widenFraction*(s_max - mu)`,
      // so the reranked (cached) set already contains everything any coverage
      // mode could keep; then stop. `widenFraction` (= f_broad + margin) sits
      // just past the floor median, so on a SMOOTH gradient the widen reranks
      // deep -- near the floor -- before the min separates (broad returns much
      // more, the cost of a wide fraction); on a CLUSTER + real background it
      // stops once the actual background min drops below the band; a genuine
      // no-match (window never separates) stops at the cap. The window is over
      // the reranked COSINE similarities (signal-agnostic -> one set serves
      // every mode/signal).
      if (reranked >= rerankCap) {
        cappedOut = reranked < n;  // capped short of the whole set
        widen = false;
      } else {
        std::vector<float> cosWindow(reranked);
        float sMax = -std::numeric_limits<float>::infinity();
        float windowMin = std::numeric_limits<float>::infinity();
        for (size_t j = 0; j < reranked; ++j) {
          const float c = 1.f - fineDists[j];
          cosWindow[j] = c;
          sMax = std::max(sMax, c);
          windowMin = std::min(windowMin, c);
        }
        const NoiseFloor f = estimateNoiseFloor(cosWindow, cut.floorFraction_);
        muMin = std::min(muMin, f.mu_);
        // Widen while the window has not yet reached below the broad band
        // `s_max - widenFraction*(s_max - muMin)`. Using the MIN mu (not this
        // batch's) means a tight top cluster (mu ~ s_max) keeps widening until
        // genuine background lowers mu; a truly flat field never separates and
        // stops at the cap (the gate / top-1 then answers).
        widen = windowMin >= sMax - cut.widenFraction_ * (sMax - muMin);
      }
    } else if (cut.mode_ == CslsCut::Mode::Softmax) {
      widen = reranked < std::min(cut.softmaxN_, n);
    } else if (batchStart == 0) {
      widen = cslsOf(batchEnd - 1, rq) >= threshold;
    } else {
      widen = false;
      for (size_t j = batchStart; j < batchEnd; ++j) {
        if (cslsOf(j, rq) >= threshold) {
          widen = true;
          break;
        }
      }
    }
    if (!widen) {
      break;
    }
    AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                << "\": csls rerank widened to "
                << std::min(reranked + floorM, n) << " of " << n
                << (plateauMode
                        ? " candidates (window not yet below the broadest band)"
                        : " candidates (cut reaches the coarse boundary)")
                << std::endl;
  }
  if (cappedOut) {
    AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                << "\": csls rerank hit the cap at " << reranked << " of " << n
                << " candidates before the window separated from the best (not "
                   "a silent truncation; raise cslsRerankFloor/the cap to go "
                   "deeper)"
                << std::endl;
  }

  // Package the reranked prefix (coarse-rank order) into the mode-independent
  // `CslsReranked`. The cut is applied separately.
  result.rq_ = rq;
  result.plateauFound_ = !cappedOut;
  result.entityBits_.resize(reranked);
  result.storeRows_.resize(reranked);
  result.fineDist_.resize(reranked);
  for (size_t j = 0; j < reranked; ++j) {
    const size_t idx = orderIdx(j);
    result.entityBits_[j] = idAt(idx);
    result.storeRows_[j] = rowAt(idx);
    result.fineDist_[j] = fineDists[j];
  }
  AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
              << "\": csls coarse+rerank: " << n << " candidates -> "
              << reranked << " reranked" << std::endl;
  return result;
}

namespace {
// A cut carries the csls sidecar requirement iff it derives a CSLS value: the
// fixed Threshold, the Knee, and the ZCut on the `csls` signal. The Softmax
// and the ZCut on the `cosine` signal are csls-independent (a plain cosine
// index suffices).
bool cutNeedsCslsSidecar(const CslsCut& cut) {
  if (cut.mode_ == CslsCut::Mode::Softmax) {
    return false;
  }
  if (cut.mode_ == CslsCut::Mode::ZCut) {
    return cut.signal_ == CslsCut::Signal::Csls;
  }
  return true;  // Threshold, Knee
}
}  // namespace

// ____________________________________________________________________________
std::vector<CslsScoredEntity> VectorIndex::searchCsls(
    ql::span<const float> query, const CslsCut& cut, size_t neighbors,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored,
    bool fullPrecision) const {
  auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.meta_.config_.csls_ || !cutNeedsCslsSidecar(cut),
                    "searchCsls called with a CSLS cut on an index without "
                    "csls data.");
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  AD_CONTRACT_CHECK(cut.mode_ != CslsCut::Mode::Softmax || cut.softmaxN_ >= 1,
                    "The softmax autoCut requires softmaxN >= 1.");
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  std::vector<char> coarseBuffer;
  const char* coarseQueryBytes =
      impl.rerank_.has_value() ? impl.scan_.encodeQuery(query, coarseBuffer)
                               : queryBytes;
  CslsReranked reranked =
      rerankCsls(impl, layer, queryBytes, coarseQueryBytes, std::nullopt, cut,
                 neighbors, candidates, checkInterrupt, fullPrecision);
  if (numScored != nullptr) {
    *numScored = reranked.scored_;
  }
  return applyCslsCutImpl(impl, reranked, cut, maxDistance);
}

// ____________________________________________________________________________
std::vector<CslsScoredEntity> VectorIndex::searchCslsByEntity(
    Id entity, const CslsCut& cut, size_t neighbors,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored,
    bool fullPrecision) const {
  auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.meta_.config_.csls_ || !cutNeedsCslsSidecar(cut),
                    "searchCslsByEntity called with a CSLS cut on an index "
                    "without csls data.");
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  AD_CONTRACT_CHECK(cut.mode_ != CslsCut::Mode::Softmax || cut.softmaxN_ >= 1,
                    "The softmax autoCut requires softmaxN >= 1.");
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    if (numScored != nullptr) {
      *numScored = 0;
    }
    return {};
  }
  CslsReranked reranked = rerankCsls(
      impl, layer, layer.rowPtr(row.value()), impl.scan_.rowPtr(row.value()),
      row, cut, neighbors, candidates, checkInterrupt, fullPrecision);
  if (numScored != nullptr) {
    *numScored = reranked.scored_;
  }
  return applyCslsCutImpl(impl, reranked, cut, maxDistance);
}

// ____________________________________________________________________________
// The MODE-INDEPENDENT cut that drives `rerankCsls`'s top-anchored widen depth.
static CslsCut plateauRerankCut(float floorFraction, float widenFraction,
                                size_t rerankCap) {
  CslsCut cut;
  cut.mode_ = CslsCut::Mode::ZCut;
  cut.floorFraction_ = floorFraction;
  cut.widenFraction_ = widenFraction;
  cut.rerankCap_ = rerankCap;
  return cut;
}

// ____________________________________________________________________________
CslsReranked VectorIndex::computeCslsReranked(
    ql::span<const float> query, size_t neighbors, float floorFraction,
    float widenFraction, size_t rerankCap,
    std::optional<ql::span<const Id>> candidates,
    const CheckInterruptCallback& checkInterrupt, bool fullPrecision) const {
  auto& impl = *impl_;
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  std::vector<char> coarseBuffer;
  const char* coarseQueryBytes =
      impl.rerank_.has_value() ? impl.scan_.encodeQuery(query, coarseBuffer)
                               : queryBytes;
  return rerankCsls(impl, layer, queryBytes, coarseQueryBytes, std::nullopt,
                    plateauRerankCut(floorFraction, widenFraction, rerankCap),
                    neighbors, candidates, checkInterrupt, fullPrecision);
}

// ____________________________________________________________________________
CslsReranked VectorIndex::computeCslsRerankedByEntity(
    Id entity, size_t neighbors, float floorFraction, float widenFraction,
    size_t rerankCap, std::optional<ql::span<const Id>> candidates,
    const CheckInterruptCallback& checkInterrupt, bool fullPrecision) const {
  auto& impl = *impl_;
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return rerankCsls(impl, layer, layer.rowPtr(row.value()),
                    impl.scan_.rowPtr(row.value()), row,
                    plateauRerankCut(floorFraction, widenFraction, rerankCap),
                    neighbors, candidates, checkInterrupt, fullPrecision);
}

// ____________________________________________________________________________
std::vector<CslsScoredEntity> VectorIndex::applyCslsCut(
    const CslsReranked& reranked, const CslsCut& cut,
    std::optional<float> maxDistance) const {
  auto& impl = *impl_;
  AD_CONTRACT_CHECK(
      impl.meta_.config_.csls_ || !cutNeedsCslsSidecar(cut),
      "applyCslsCut called with a CSLS cut on an index without csls data.");
  AD_CONTRACT_CHECK(cut.mode_ != CslsCut::Mode::Softmax || cut.softmaxN_ >= 1,
                    "The softmax autoCut requires softmaxN >= 1.");
  return applyCslsCutImpl(impl, reranked, cut, maxDistance);
}

}  // namespace qlever::vector
