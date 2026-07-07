// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H

#include <cstdint>
#include <stdexcept>
#include <string>

#include "util/json.h"

// On-disk format description and shared types for the built-in vector indices.
//
// A single named vector index `N` of a database with base name `B` is stored in
// the following files:
//
//   B.vec.N.meta    JSON metadata (this file's `VectorIndexMetadata`)
//   B.vec.N.data    flat row-major matrix in the configured scalar type
//                   (f32/f16/i8), stride = `dimensions` scalars. IMMUTABLE
//                   after the build: row indices are the permanent identity of
//                   the vectors.
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
//     equals the natural row byte length `dimensions * bytesPerScalar`, so v5
//     is otherwise identical to v4. (Historically v5 also allowed opt-in
//     64-byte row padding; that build option was removed and the stride is now
//     always the natural length.) Version 4 indices still load unchanged
//     (their stride is derived).
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
// rejects `i8` + `l2sq`/`innerProduct`). (`Bf16` is appended after `I8` to
// keep the preexisting enum values stable.)
enum class VectorScalar : uint8_t { F32, F16, I8, Bf16 };

// Bytes per stored scalar value.
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
  }
  return 4;
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
  }
  return "f32";
}

inline VectorScalar vectorScalarFromString(std::string_view s) {
  if (s == "f32") return VectorScalar::F32;
  if (s == "f16") return VectorScalar::F16;
  if (s == "i8") return VectorScalar::I8;
  if (s == "bf16") return VectorScalar::Bf16;
  throw std::runtime_error("Unknown vector scalar type: " + std::string{s});
}

// Build-time configuration + identity of one named vector index.
struct VectorIndexConfig {
  std::string name_;
  uint32_t dimensions_ = 0;
  VectorMetric metric_ = VectorMetric::Cosine;
  VectorScalar scalar_ = VectorScalar::F32;
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
  // byte length `dimensions * bytesPerScalar`. 0 means "not stored" (a v4
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
inline std::string vectorIrisFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".iris";
}
inline std::string vectorHnswFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".hnsw";
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
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H
