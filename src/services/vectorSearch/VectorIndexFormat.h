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
//   B.vec.N.data    flat row-major float matrix, stride = `dimensions` floats.
//                   IMMUTABLE after the build: row indices are the permanent
//                   identity of the vectors.
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
// `.rowmap` sidecars, tombstones (cheap remapping after a KG re-index).
inline constexpr uint32_t VECTOR_INDEX_VERSION = 3;

// The `.keys` value of a row whose entity is not part of the current knowledge
// graph. Real keys are `ValueId` bits, whose datatype (high) bits never have
// all bits set, so this value cannot collide.
inline constexpr uint64_t TOMBSTONE_KEY = ~uint64_t{0};

// One entry of the `.rowmap` file: the id -> row direction of the entity
// mapping, sorted by `idBits_`.
struct IdRowPair {
  uint64_t idBits_;
  uint64_t row_;
};

// The similarity metric of an index. Maps 1:1 to a `usearch` metric kind.
enum class VectorMetric : uint8_t { Cosine, L2Sq, InnerProduct };

// The scalar type the vectors are stored as. `F32` is the default; the smaller
// types trade recall for size (and are added in a later phase).
enum class VectorScalar : uint8_t { F32, F16, I8 };

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
  }
  return "f32";
}

inline VectorScalar vectorScalarFromString(std::string_view s) {
  if (s == "f32") return VectorScalar::F32;
  if (s == "f16") return VectorScalar::F16;
  if (s == "i8") return VectorScalar::I8;
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

  // Optional embedding endpoint, bound to this index so that query-time
  // embedding always uses the SAME model that produced the index.
  // `embeddingUrl_` is an OpenAI-compatible base URL (the client appends
  // `/v1/embeddings`); empty means the index is vector-only (no
  // `vec:queryText`).
  std::string embeddingUrl_;
  std::string embeddingModel_;
};

// One vector index to build during `qlever index`: its configuration plus the
// input files (currently a `.npy` float matrix + a row-aligned IRI list; a
// Parquet reader is planned). Each input IRI is resolved against the freshly
// built knowledge-graph vocabulary; rows whose IRI is unknown are skipped.
struct VectorIndexBuildSpec {
  VectorIndexConfig config_;
  std::string irisPath_;
  // Exactly one source of vectors:
  std::string npyPath_;    // precomputed vectors in a .npy matrix, or
  std::string textsPath_;  // a row-aligned file of texts to embed at index time
                           // (requires `config_.embeddingUrl_`).
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
                     {"embeddingUrl", m.config_.embeddingUrl_},
                     {"embeddingModel", m.config_.embeddingModel_},
                     {"version", m.version_},
                     {"vocabSize", m.vocabSize_}};
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
  m.config_.embeddingUrl_ = j.value("embeddingUrl", std::string{});
  m.config_.embeddingModel_ = j.value("embeddingModel", std::string{});
  j.at("version").get_to(m.version_);
  m.vocabSize_ = j.value("vocabSize", uint64_t{0});
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXFORMAT_H
