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
// the following files (mirroring how the text index uses `B.text.*`):
//
//   B.vec.N.meta   JSON metadata (this file's `VectorIndexMetadata`)
//   B.vec.N.keys   `MmapVector<uint64_t>`: ascending entity ids (VocabIndex)
//   B.vec.N.data   flat row-major float matrix, stride = `dimensions` floats
//   B.vec.N.hnsw   usearch HNSW index file (only if `hasHnsw_`)
//
// Row `i` of `.data` is the vector of the entity whose id is `keys[i]`. Because
// `.keys` is sorted, the entity -> row mapping is a binary search and row ->
// entity is O(1). Only entities that actually have a vector occupy a row, so a
// sparse index over a large database stays small.
namespace qlever::vector {

// Bumped whenever the on-disk layout of the `.keys`/`.data`/`.meta` files
// changes. Independent of QLever's global `indexFormatVersion`, so adding or
// changing vector indices never forces a rebuild of the main index.
// Version history: 1 = initial; 2 = added the `vocabSize` fingerprint.
inline constexpr uint32_t VECTOR_INDEX_VERSION = 2;

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
};

// What is persisted in `B.vec.N.meta` and re-read at server start.
struct VectorIndexMetadata {
  VectorIndexConfig config_;
  uint64_t numVectors_ = 0;
  bool hasHnsw_ = false;
  uint32_t version_ = VECTOR_INDEX_VERSION;
  // Fingerprint of the knowledge-graph build the entity ids in `.keys` refer
  // to: the size of the vocabulary at vector-index build time. The stored ids
  // are vocabulary positions, so they silently point at DIFFERENT entities
  // after the main index is rebuilt with changed data; the load hook compares
  // this fingerprint against the loaded vocabulary and skips (with a warning)
  // any vector index that does not match.
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
inline std::string vectorDataFile(const std::string& base,
                                  const std::string& name) {
  return base + ".vec." + name + ".data";
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
