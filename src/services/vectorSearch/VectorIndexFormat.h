// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "util/json.h"

// On-disk format description and shared types for the built-in vector indices.
//
// A single named vector index `N` of a database with base name `B` is stored in
// the following files:
//
//   B.vec.N.meta    JSON metadata (this file's `VectorIndexMetadata`)
//   B.vec.N.data    flat row-major matrix in the configured scalar type
//                   (f32/f16/bf16/i8, or 1-bit sign-packed `binary`), row
//                   length = `rowBytesFor(scalar, dimensions)`. IMMUTABLE
//                   after the build: row indices are the permanent identity of
//                   the vectors.
//   B.vec.N.rerank.data
//                   OPTIONAL second flat matrix (only with the two-layer
//                   quantize+rerank build, `rerankScalar_`): the SAME rows in
//                   the SAME order as `.data`, but stored at the (higher)
//                   rerank precision (bf16/f16/f32). `.data` is then the
//                   coarse SCAN layer (e.g. i8) that brute-force scans and the
//                   HNSW graph read; this file is the fine RERANK layer that
//                   `vec:distance` and the SERVICE's rerank pass read.
//                   IMMUTABLE after the build, natural (unpadded) row stride.
//   B.vec.N.iris    one IRI per line, aligned with the rows. IMMUTABLE. Kept so
//                   that the entity mapping can be recomputed cheaply after the
//                   knowledge graph is re-indexed (see "remapping" below).
//   B.vec.N.keys    `MmapVector<uint64_t>`: row -> current `ValueId` bits, or
//                   `TOMBSTONE_KEY` for rows whose IRI is not part of the
//                   current knowledge graph. Rewritten by a remap.
//   B.vec.N.rowmap  `MmapVector<IdRowPair>` sorted by id: current `ValueId`
//                   bits -> row, live rows only (the id -> row direction).
//                   Rewritten by a remap.
//   B.vec.N.hnsw    usearch HNSW graph (only if `hasHnsw_`), keyed by ROW
//                   index; the vectors themselves are NOT duplicated into this
//                   file -- distance computations read them from `.data`.
//                   IMMUTABLE after the build.
//   B.vec.N.csls    OPTIONAL (only with `csls: true`): `numVectors` f32 in ROW
//                   order -- the per-document CSLS hubness `r(d)`, the mean
//                   cosine similarity of each row to its `cslsNeighbors`
//                   nearest corpus neighbours (self-excluded). Row-aligned
//                   like `.data`, so it survives a remap unchanged. IMMUTABLE
//                   after the build.
//
// Remapping: the entity ids stored in `.keys`/`.rowmap` are vocabulary
// positions, which shift whenever the RDF data is re-indexed. Because `.data`,
// `.iris`, and the row-keyed `.hnsw` never depend on those ids, a re-index of
// the RDF data only requires re-resolving `.iris` against the new vocabulary
// and rewriting the two small mapping files -- the (potentially huge) vector
// matrix and HNSW graph are reused as-is. Entities that disappeared from the
// knowledge graph become tombstones (skipped by all searches).
namespace qlever::vector {

// Bumped whenever the on-disk layout changes. Independent of QLever's global
// `indexFormatVersion`, so adding or changing vector indices never forces a
// rebuild of the main index. Version history: 1 = initial; 2 = added the
// `vocabSize` fingerprint; 3 = row-keyed graph-only `.hnsw`, `.iris` +
// `.rowmap` sidecars, tombstones (cheap remapping after a KG re-index);
// 4 = `.data` holds the configured scalar type (f32/f16/i8) as raw bytes;
// 5 = added an explicit per-row byte STRIDE (`rowStrideBytes_`); it always
//     equals the natural row byte length `rowBytesFor(scalar, dimensions)`
//     (`dimensions * bytesPerScalar`; `(dimensions + 7) / 8` for the packed
//     1-bit `binary` scalar), so v5
//     is otherwise identical to v4. (Historically v5 also allowed opt-in
//     64-byte row padding; that build option was removed and the stride is now
//     always the natural length.) Version 4 indices still load unchanged
//     (their stride is derived). A v5 index may ADDITIONALLY carry the
//     optional two-layer rerank matrix (`rerankScalar_` + `.rerank.data`);
//     a `.meta` without the field is a plain single-layer index, so old
//     indices load unchanged and no version bump is needed.
inline constexpr uint32_t VECTOR_INDEX_VERSION = 5;

// The set of on-disk versions this binary can still read. A v4 index (no
// explicit stride) loads with `rowStrideBytes_` derived as `dimensions *
// bytesPerScalar`, so it never needs a forced rebuild.
inline bool isSupportedVectorIndexVersion(uint32_t version) {
  return version == 4 || version == VECTOR_INDEX_VERSION;
}

// The `.keys` value of a row whose entity is not part of the current knowledge
// graph. Real keys are `ValueId` bits, whose datatype (high) bits never have
// all bits set, so this value cannot collide.
inline constexpr uint64_t TOMBSTONE_KEY = ~uint64_t{0};

// A sanity ceiling on the vector dimension. Real embedding models are far
// below this; the bound prevents a crafted/typo'd input from requesting a
// multi-gigabyte per-vector allocation.
inline constexpr uint32_t MAX_VECTOR_DIMENSIONS = 1u << 16;  // 65536

// One entry of the `.rowmap` file: the id -> row direction of the entity
// mapping, sorted by `idBits_`.
struct IdRowPair {
  uint64_t idBits_;
  uint64_t row_;
};

// Two-pointer merge join of an ascending-by-id candidate id range against the
// ascending-by-id `.rowmap`. Calls `emit(row, idBits)` once for every candidate
// id that has a live row (duplicate candidate ids are collapsed). Both
// `[candFirst, candLast)` and `[rmFirst, rmLast)` MUST be sorted ascending by
// id bits. Runs in O(#candidates + #rows).
//
// Because the flat store is stored in id (== IRI-lex) order, the rowmap's
// `row_` values are monotonically non-decreasing in id, so the rows this emits
// come out non-decreasing too => sequential `.data` reads. IMPORTANT:
// correctness does NOT depend on that invariant -- a merge over the same id key
// is correct regardless; only the *sequentiality* of the emitted rows (and thus
// the read-ahead win) relies on the store being id-sorted. A candidate id that
// is not a stored entity (e.g. a local-vocab or non-`VocabIndex` id) simply
// fails to match and is skipped.
template <typename CandIt, typename RowmapIt, typename Emit>
void mergeJoinRowmap(CandIt candFirst, CandIt candLast, RowmapIt rmFirst,
                     RowmapIt rmLast, Emit&& emit) {
  while (candFirst != candLast && rmFirst != rmLast) {
    uint64_t candBits = *candFirst;
    uint64_t rowBits = rmFirst->idBits_;
    if (candBits < rowBits) {
      ++candFirst;
    } else if (rowBits < candBits) {
      ++rmFirst;
    } else {
      emit(static_cast<size_t>(rmFirst->row_), candBits);
      ++candFirst;
      ++rmFirst;
    }
  }
}

// The similarity metric of an index. Maps 1:1 to a `usearch` metric kind.
enum class VectorMetric : uint8_t { Cosine, L2Sq, InnerProduct };

// The scalar type the vectors are stored as. `F32` is the default; the
// smaller types trade a little recall for a half/quarter storage (and page
// cache) footprint. `Bf16` is the lossless choice for embeddings that were
// produced in bf16: the fp32 `.npy` input is a bf16 upscaled to f32, and
// truncating it back to bf16 recovers the original bits exactly. NOTE: `I8`
// rescales every vector to a common magnitude (usearch's dot-product-oriented
// int8 cast), so it is only meaningful with the `cosine` metric (the builder
// rejects `i8` + `l2sq`/`innerProduct`). `Binary` is the 1-BIT rung: only the
// SIGN of each component is kept (bit i set iff component i > 0, packed 8 per
// byte -- a row is `(dim + 7) / 8` bytes, see `rowBytesFor`), and rows are
// compared by HAMMING distance (differing sign bits), an angular proxy --
// like `i8` it is cosine-only, and it is meant as the coarse SCAN layer of a
// two-layer index whose fine layer rescores exactly. (Values are appended in
// the order the scalars were introduced to keep the preexisting enum values
// stable.)
enum class VectorScalar : uint8_t { F32, F16, I8, Bf16, Binary };

// Which per-row/block kernel to use for the exact bf16 COSINE distance of the
// fine layer (the `vec:bf16Kernel` query knob). Purely a performance A/B dial:
// every value computes the SAME cosine distance (to the documented ~1e-6
// tolerance across kernels), so it never changes the result set for distinct
// distances. `Auto` (the default) picks the fastest kernel the running CPU
// supports; the explicit values force one, and are silently downgraded to a
// supported one (never a hard error) when the CPU lacks the ISA:
//   - `Amx`    the fixed width-1 AMX-BF16 tile GEMM (Sapphire Rapids+).
//   - `Simd`   the multi-row AVX-512-BF16 `vdpbf16ps` kernel (Genoa/SPR+).
//   - `Punned` usearch's per-pair NumKong metric (the portable fallback).
// A no-op on any non-bf16 / non-cosine layer (those always use the punned
// metric). NOT persisted -- a per-query serving choice only.
enum class Bf16Kernel : uint8_t { Auto, Amx, Simd, Punned };

// Bytes per stored scalar value. UNDEFINED for `Binary` (a scalar is a single
// bit, 8 packed per byte) -- every row/byte-length computation must go through
// `rowBytesFor` below, which handles the packed-bit case; calling this with
// `Binary` throws so a missed call site fails loudly instead of sizing a
// matrix with 0-byte rows.
inline size_t bytesPerScalar(VectorScalar s) {
  switch (s) {
    case VectorScalar::F32:
      return 4;
    case VectorScalar::F16:
      return 2;
    case VectorScalar::I8:
      return 1;
    case VectorScalar::Bf16:
      return 2;
    case VectorScalar::Binary:
      throw std::runtime_error(
          "bytesPerScalar is undefined for the sub-byte `binary` scalar; "
          "compute row lengths via rowBytesFor.");
  }
  return 4;
}

// Byte length of one row of `dim` scalars of type `s`: `dim` whole scalars,
// except for the packed 1-bit `Binary` rows (8 sign bits per byte, the last
// byte's unused bits zero-padded). The ONLY way row byte lengths may be
// computed -- `dim * bytesPerScalar(s)` is wrong (and throws) for `Binary`.
inline size_t rowBytesFor(VectorScalar s, size_t dim) {
  return s == VectorScalar::Binary ? (dim + 7) / 8 : dim * bytesPerScalar(s);
}

// Default coarse-candidate count of the two-layer rerank pass when
// `vec:rerankK` is unset (always additionally clamped to >= k by the
// callers). The `binary` scan layer keeps 1 bit per component (vs i8's 8), so
// its Hamming pre-ranking is far coarser and needs a wider candidate margin
// for the fine rerank to recover the true top-k.
inline size_t defaultRerankK(VectorScalar scanScalar, size_t k) {
  return scanScalar == VectorScalar::Binary ? std::max<size_t>(50 * k, 500)
                                            : std::max<size_t>(10 * k, 100);
}

// The CSLS r-term primitive, shared by the build-time r(d) and the query-time
// r(q): keep the `k` SMALLEST cosine DISTANCES offered, and average them as
// cosine SIMILARITIES via `cos_sim = 1 - distance` (the usearch/NumKong
// angular-distance convention -- CSLS works in similarity space).
class CslsNeighborhood {
 public:
  explicit CslsNeighborhood(size_t k) : k_{k} {}
  void offer(float distance) {
    if (k_ == 0) {
      return;
    }
    if (heap_.size() < k_) {
      heap_.push_back(distance);
      std::push_heap(heap_.begin(), heap_.end());
    } else if (distance < heap_.front()) {
      std::pop_heap(heap_.begin(), heap_.end());
      heap_.back() = distance;
      std::push_heap(heap_.begin(), heap_.end());
    }
  }
  // Mean cosine similarity of the kept neighbours; fewer than `k` neighbours
  // average what there is, no neighbours at all (a lone vector) -> 0.
  float meanCosSim() const {
    if (heap_.empty()) {
      return 0.f;
    }
    double sum = 0;
    for (float d : heap_) {
      sum += 1.0 - static_cast<double>(d);
    }
    return static_cast<float>(sum / static_cast<double>(heap_.size()));
  }

  // Population standard deviation of the kept neighbours' cosine similarities --
  // the local spread of a point's nearest neighbours. Used at BUILD time to
  // calibrate the softmax autoCut temperature: a tightly-clustered space (small
  // spread) wants a small, sharp T; a diffuse one a larger T. Fewer than two
  // neighbours -> 0 (no spread to measure).
  float stdevCosSim() const {
    if (heap_.size() < 2) {
      return 0.f;
    }
    const double mean = static_cast<double>(meanCosSim());
    double sumSq = 0;
    for (float d : heap_) {
      const double c = (1.0 - static_cast<double>(d)) - mean;
      sumSq += c * c;
    }
    return static_cast<float>(
        std::sqrt(sumSq / static_cast<double>(heap_.size())));
  }

 private:
  size_t k_;
  std::vector<float> heap_;  // max-heap of the k smallest distances
};

// Distribution summary of a csls r(d) array via a fixed-size STACK histogram --
// NO large heap copy (unlike copy+sort), so it is immune to any transient
// heap corruption in the builder process. That corruption is precisely what
// made the build-time `logCslsDistribution` spuriously report 1/1/1 (it sorted
// an 8.5 MB heap copy that got stomped) even though the persisted, file-backed
// sidecar -- and hence the runtime -- was correct all along. r(d) is a cosine
// mean in [0,1]; min/max are exact, percentiles are to ~1/kBuckets resolution.
struct CslsRdSummary {
  float min_ = 0.f, p50_ = 0.f, p95_ = 0.f, max_ = 0.f;
  double mean_ = 0.0;
};
inline CslsRdSummary summarizeCslsRd(ql::span<const float> r) {
  CslsRdSummary s;
  if (r.empty()) {
    return s;
  }
  constexpr size_t kBuckets = 4096;
  std::array<uint64_t, kBuckets> hist{};  // 32 KiB on the STACK, no heap alloc
  float mn = r[0], mx = r[0];
  double sum = 0.0;
  for (float v : r) {
    mn = std::min(mn, v);
    mx = std::max(mx, v);
    sum += static_cast<double>(v);
    const float c = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    ++hist[std::min(kBuckets - 1, static_cast<size_t>(c * kBuckets))];
  }
  auto percentile = [&](double frac) {
    const auto target =
        static_cast<uint64_t>(frac * static_cast<double>(r.size()));
    uint64_t cum = 0;
    for (size_t b = 0; b < kBuckets; ++b) {
      cum += hist[b];
      if (cum > target) {
        return static_cast<float>(b) / static_cast<float>(kBuckets);
      }
    }
    return 1.f;
  };
  s.min_ = mn;
  s.max_ = mx;
  s.mean_ = sum / static_cast<double>(r.size());
  s.p50_ = percentile(0.5);
  s.p95_ = percentile(0.95);
  return s;
}

// String conversions (used both for the SPARQL surface and the JSON metadata).
inline std::string toString(VectorMetric m) {
  switch (m) {
    case VectorMetric::Cosine:
      return "cosine";
    case VectorMetric::L2Sq:
      return "l2sq";
    case VectorMetric::InnerProduct:
      return "innerProduct";
  }
  return "cosine";
}

inline VectorMetric vectorMetricFromString(std::string_view s) {
  if (s == "cosine") return VectorMetric::Cosine;
  if (s == "l2sq" || s == "l2") return VectorMetric::L2Sq;
  if (s == "innerProduct" || s == "ip" || s == "dot")
    return VectorMetric::InnerProduct;
  throw std::runtime_error("Unknown vector metric: " + std::string{s});
}

inline std::string toString(VectorScalar s) {
  switch (s) {
    case VectorScalar::F32:
      return "f32";
    case VectorScalar::F16:
      return "f16";
    case VectorScalar::I8:
      return "i8";
    case VectorScalar::Bf16:
      return "bf16";
    case VectorScalar::Binary:
      return "binary";
  }
  return "f32";
}

inline VectorScalar vectorScalarFromString(std::string_view s) {
  if (s == "f32") return VectorScalar::F32;
  if (s == "f16") return VectorScalar::F16;
  if (s == "i8") return VectorScalar::I8;
  if (s == "bf16") return VectorScalar::Bf16;
  if (s == "binary") return VectorScalar::Binary;
  throw std::runtime_error("Unknown vector scalar type: " + std::string{s});
}

// Build-time configuration + identity of one named vector index.
struct VectorIndexConfig {
  std::string name_;
  uint32_t dimensions_ = 0;
  VectorMetric metric_ = VectorMetric::Cosine;
  VectorScalar scalar_ = VectorScalar::F32;
  // Two-layer quantize+rerank build: when set, the builder writes -- from the
  // SAME f32 input -- BOTH the coarse scan matrix `.data` (in `scalar_`, e.g.
  // i8 or binary) AND a second fine matrix `.rerank.data` at this precision
  // (bf16, f16, or f32; NEVER i8/binary -- the rerank layer is the
  // high-precision one). The fine layer carries the same metric. `nullopt`
  // (the default) = single-layer, exactly as before. Storage cost = scan
  // bytes + rerank bytes (e.g. i8 + bf16 = 3 bytes per dimension; binary +
  // bf16 = 2.125 bytes per dimension).
  std::optional<VectorScalar> rerankScalar_;
  // CSLS (Cross-domain Similarity Local Scaling) hub suppression: when true,
  // the build additionally computes -- or ingests, see `cslsRPath_` -- the
  // per-document hubness `r(d)` = the mean COSINE SIMILARITY of each stored
  // vector to its `cslsNeighbors_` nearest corpus neighbours (self-excluded),
  // persisted as the row-aligned f32 sidecar `.csls`. At query time
  // `vec:cslsThreshold` then keeps a candidate `d` iff
  // `2 * cos_sim(q, d) - r(q) - r(d) >= threshold` -- a query-adaptive
  // "stand-out" cut that replaces a hardcoded top-k and penalises hub
  // documents that are near everything. Cosine-only: `cos_sim = 1 - distance`
  // holds only for the cosine metric (usearch/NumKong angular distance), so
  // the builder rejects `csls` for `l2sq`/`innerProduct` (and for a `binary`
  // store without a rerank layer, whose only distances are Hamming).
  bool csls_ = false;
  // The neighbour count k of the CSLS r-terms: both the build-time r(d) and
  // the query-time r(q) average the top-k cosine similarities. Persisted in
  // the `.meta` (only when `csls_` is set).
  uint32_t cslsNeighbors_ = 10;
  // Optional path to a PRECOMPUTED r(d) as an `.npy` of f32 (shape `(N,)` or
  // `(N, 1)`), row-aligned with the `npy` input matrix -- the "GPU path" for
  // corpora where the build-time self-kNN is too slow. When set, the build
  // ingests the values verbatim (following the input rows through the usual
  // skip/dedup) instead of computing the self-kNN. Build-time only, never
  // persisted (like `buildThreads_`).
  std::string cslsRPath_;
  bool buildHnsw_ = true;
  // HNSW (usearch) build parameters.
  uint32_t hnswConnectivity_ = 16;     // a.k.a. M
  uint32_t hnswExpansionAdd_ = 128;    // a.k.a. efConstruction
  uint32_t hnswExpansionSearch_ = 64;  // a.k.a. efSearch
  // Number of threads for the HNSW construction; 0 = all hardware threads.
  // Build-time only (not persisted).
  uint32_t buildThreads_ = 0;

  // RAM-residency preference for the flat store, applied by the loader at
  // `open()`: "none" (mmap, paged on demand), "advise" (MADV_WILLNEED
  // prefault), "lock" (also mlock -- fault-free, non-evictable), or "aligned"
  // (a huge-page-backed, 64-byte-aligned RAM copy). Gated on fits-in-RAM at
  // load time. NOT persisted: this is a serving concern, set solely IN MEMORY
  // at server start from the per-index "preload" field of the
  // `QLEVER_VECTOR_SEARCH_ENDPOINTS` environment variable (see
  // `VectorIndexExtension.h`), which the load hook threads into
  // `open(..., residency)`. Defaults to "none" and stays "none" when unset.
  std::string preload_ = "none";

  // Query-time embedding endpoint of this index (what `vec:embed` POSTs to,
  // and the model identity stamped into the typed query-vector literal).
  // `embeddingUrl_` is an OpenAI-compatible base URL (the client appends
  // `/v1/embeddings`); empty means the index is vector-only (no `vec:embed`).
  // NOT persisted: like `preload_`, a serving concern set solely IN MEMORY at
  // server start from `QLEVER_VECTOR_SEARCH_ENDPOINTS` (see
  // `setEmbeddingEndpoint`). Both default to empty.
  std::string embeddingUrl_;
  std::string embeddingModel_;
};

// One vector index to build during `qlever index`: its configuration plus the
// input files -- a precomputed `.npy` float matrix and a row-aligned IRI list.
// Each input IRI is resolved against the freshly built knowledge-graph
// vocabulary; rows whose IRI is unknown are skipped.
struct VectorIndexBuildSpec {
  VectorIndexConfig config_;
  // Row-aligned entity-IRI list, one per matrix row.
  std::string irisPath_;
  // The precomputed vectors as an N x D `.npy` matrix.
  std::string npyPath_;
  // If true, do not build anything: re-resolve the existing index's `.iris`
  // against the (re-indexed) knowledge graph and rewrite `.keys`/`.rowmap`.
  bool remap_ = false;
};

// What is persisted in `B.vec.N.meta` and re-read at server start.
struct VectorIndexMetadata {
  VectorIndexConfig config_;
  // Number of rows in `.data` (including tombstoned rows).
  uint64_t numVectors_ = 0;
  // Number of rows whose entity is absent from the current knowledge graph.
  uint64_t numTombstones_ = 0;
  bool hasHnsw_ = false;
  uint32_t version_ = VECTOR_INDEX_VERSION;
  // Fingerprint of the knowledge-graph build the entity ids in
  // `.keys`/`.rowmap` refer to: the size of the vocabulary at (re)mapping
  // time. The load hook compares this fingerprint against the loaded
  // vocabulary and skips (with a warning suggesting a remap) any vector index
  // that does not match.
  uint64_t vocabSize_ = 0;
  // Byte stride between consecutive rows in `.data`. Always the natural row
  // byte length `rowBytesFor(scalar, dimensions)`. 0 means "not stored" (a v4
  // index), in which case the loader derives it as the raw row byte length.
  uint64_t rowStrideBytes_ = 0;
  // Fingerprint of the knowledge-graph vocabulary's collation
  // (`LocaleManager::getCollationIdentifier()`) at (re)mapping time. The
  // store's rows are laid out in vocabulary (collation) order; a changed
  // collation would make that order stale relative to the current ids. Empty
  // means "not stored" (a v4 index, or a build before this field existed).
  // Purely a guard: the load hook WARNs on a mismatch but keeps serving
  // (candidate gathers stay correct, they merely lose the sequential-read
  // speedup). Never affects results.
  std::string collationLocale_;
  // Softmax-autoCut temperature T calibrated from the corpus at build time: the
  // median, across rows, of each point's nearest-neighbour cosine spread
  // (`CslsNeighborhood::stdevCosSim`), clamped to a sane range. A saturated
  // space (cosines packed near 1) yields a small, sharp T automatically. Set
  // only for a csls index whose r(d) was COMPUTED (not ingested via `cslsR`);
  // absent => the query path falls back to the constant default. A SERVING
  // default only: the runtime config and the per-query `vec:softmaxTemperature`
  // both override it (see `resolveCslsCut`).
  std::optional<float> calibratedSoftmaxT_;
};

// Path helpers. Centralised so the builder and the reader agree.
inline std::string vectorMetaFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".meta";
}
inline std::string vectorKeysFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".keys";
}
inline std::string vectorRowmapFile(const std::string& base,
                                    const std::string& name) {
  return base + ".vec." + name + ".rowmap";
}
inline std::string vectorDataFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".data";
}
inline std::string vectorRerankDataFile(const std::string& base,
                                        const std::string& name) {
  return base + ".vec." + name + ".rerank.data";
}
inline std::string vectorIrisFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".iris";
}
inline std::string vectorHnswFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".hnsw";
}
inline std::string vectorCslsFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".csls";
}

// JSON (de)serialization of the metadata (uses QLever's bundled
// nlohmann::json).
inline void to_json(nlohmann::json& j, const VectorIndexMetadata& m) {
  j = nlohmann::json{{"name", m.config_.name_},
                     {"dimensions", m.config_.dimensions_},
                     {"metric", toString(m.config_.metric_)},
                     {"scalar", toString(m.config_.scalar_)},
                     {"numVectors", m.numVectors_},
                     {"numTombstones", m.numTombstones_},
                     {"hasHnsw", m.hasHnsw_},
                     {"hnswConnectivity", m.config_.hnswConnectivity_},
                     {"hnswExpansionAdd", m.config_.hnswExpansionAdd_},
                     {"hnswExpansionSearch", m.config_.hnswExpansionSearch_},
                     {"version", m.version_},
                     {"vocabSize", m.vocabSize_},
                     {"rowStrideBytes", m.rowStrideBytes_},
                     {"collationLocale", m.collationLocale_}};
  // The two-layer rerank precision is only written when present, so a
  // single-layer `.meta` stays byte-compatible with older builds.
  if (m.config_.rerankScalar_.has_value()) {
    j["rerankScalar"] = toString(m.config_.rerankScalar_.value());
  }
  // The CSLS neighbour count is only written for a csls-enabled index; its
  // PRESENCE is what marks the index as csls-enabled at load (`from_json`), so
  // a non-csls `.meta` stays byte-compatible with older builds. (`cslsRPath_`
  // is a build-time input path, deliberately not persisted.)
  if (m.config_.csls_) {
    j["cslsNeighbors"] = m.config_.cslsNeighbors_;
  }
  // Build-time softmax T calibration: only written when it was computed (a csls
  // index with a self-kNN r(d)), so it stays absent -- and back-compatible --
  // otherwise.
  if (m.calibratedSoftmaxT_.has_value()) {
    j["calibratedSoftmaxT"] = m.calibratedSoftmaxT_.value();
  }
  // NOTE: `embeddingUrl`/`embeddingModel`/`preload` are deliberately NOT
  // persisted -- they are serving concerns set at server start from the
  // `QLEVER_VECTOR_SEARCH_ENDPOINTS` environment variable (see
  // `VectorIndexExtension.h`). Older `.meta` files may still carry those keys;
  // `from_json` simply ignores them.
}

inline void from_json(const nlohmann::json& j, VectorIndexMetadata& m) {
  j.at("name").get_to(m.config_.name_);
  j.at("dimensions").get_to(m.config_.dimensions_);
  m.config_.metric_ = vectorMetricFromString(j.at("metric").get<std::string>());
  m.config_.scalar_ = vectorScalarFromString(j.at("scalar").get<std::string>());
  // Absent (the back-compat default) = single-layer, exactly the old format.
  if (std::string rerank = j.value("rerankScalar", std::string{});
      !rerank.empty()) {
    m.config_.rerankScalar_ = vectorScalarFromString(rerank);
  } else {
    m.config_.rerankScalar_ = std::nullopt;
  }
  // Absent (the back-compat default) = no csls sidecar, exactly the old
  // format; present = csls-enabled with that neighbour count (see `to_json`).
  m.config_.csls_ = j.contains("cslsNeighbors");
  m.config_.cslsNeighbors_ = j.value("cslsNeighbors", uint32_t{10});
  j.at("numVectors").get_to(m.numVectors_);
  m.numTombstones_ = j.value("numTombstones", uint64_t{0});
  j.at("hasHnsw").get_to(m.hasHnsw_);
  j.at("hnswConnectivity").get_to(m.config_.hnswConnectivity_);
  j.at("hnswExpansionAdd").get_to(m.config_.hnswExpansionAdd_);
  j.at("hnswExpansionSearch").get_to(m.config_.hnswExpansionSearch_);
  j.at("version").get_to(m.version_);
  m.vocabSize_ = j.value("vocabSize", uint64_t{0});
  // Absent in v4 (0 => the loader derives the raw row byte length).
  m.rowStrideBytes_ = j.value("rowStrideBytes", uint64_t{0});
  // `embeddingUrl`/`embeddingModel`/`preload` are intentionally NOT read: they
  // are serving concerns set at server start (see `to_json`). Older `.meta`
  // files that still carry those keys load fine -- the extra keys are ignored,
  // and the in-memory fields keep their defaults (empty / "none").
  // Absent before the collation guard existed ("" => the load hook skips the
  // collation check for this index).
  m.collationLocale_ = j.value("collationLocale", std::string{});
  // Absent (older builds, ingested-r(d), or non-csls) => no calibrated default;
  // the query path then uses the constant softmax temperature.
  if (j.contains("calibratedSoftmaxT")) {
    m.calibratedSoftmaxT_ = j.at("calibratedSoftmaxT").get<float>();
  } else {
    m.calibratedSoftmaxT_ = std::nullopt;
  }
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H
