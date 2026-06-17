// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#ifndef QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXFORMAT_H
#define QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXFORMAT_H

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
//   B.vec.N.hnsw   usearch HNSW index file (only if `hasHnsw`)
//
// Row `i` of `.data` is the vector of the entity whose id is `keys[i]`. Because
// `.keys` is sorted, the entity -> row mapping is a binary search and row ->
// entity is O(1). Only entities that actually have a vector occupy a row, so a
// sparse index over a large database stays small.
namespace qlever::vector {

// Bumped whenever the on-disk layout of the `.keys`/`.data`/`.meta` files
// changes. Independent of QLever's global `indexFormatVersion`, so adding or
// changing vector indices never forces a rebuild of the main index.
inline constexpr uint32_t VECTOR_INDEX_VERSION = 1;

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
  std::string name;
  uint32_t dimensions = 0;
  VectorMetric metric = VectorMetric::Cosine;
  VectorScalar scalar = VectorScalar::F32;
  bool buildHnsw = true;
  // HNSW (usearch) build parameters.
  uint32_t hnswConnectivity = 16;       // a.k.a. M
  uint32_t hnswExpansionAdd = 128;      // a.k.a. efConstruction
  uint32_t hnswExpansionSearch = 64;    // a.k.a. efSearch

  // Optional embedding endpoint, bound to this index so that query-time
  // embedding always uses the SAME model that produced the index (see
  // `docs/vector-index/embedding-generation.md`). `embeddingUrl` is an
  // OpenAI-compatible base URL (the client appends `/v1/embeddings`); empty
  // means the index is vector-only (no `vec:queryText`).
  std::string embeddingUrl;
  std::string embeddingModel;
};

// One vector index to build during `qlever index`: its configuration plus the
// input files (currently a `.npy` float matrix + a row-aligned IRI list; a
// Parquet reader is planned). Each input IRI is resolved against the freshly
// built knowledge-graph vocabulary; rows whose IRI is unknown are skipped.
struct VectorIndexBuildSpec {
  VectorIndexConfig config;
  std::string irisPath;
  // Exactly one source of vectors:
  std::string npyPath;    // precomputed vectors in a .npy matrix, or
  std::string textsPath;  // a row-aligned file of texts to embed at index time
                          // (requires `config.embeddingUrl`).
};

// What is persisted in `B.vec.N.meta` and re-read at server start.
struct VectorIndexMetadata {
  VectorIndexConfig config;
  uint64_t numVectors = 0;
  bool hasHnsw = false;
  uint32_t version = VECTOR_INDEX_VERSION;
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

// JSON (de)serialization of the metadata (uses QLever's bundled nlohmann::json).
inline void to_json(nlohmann::json& j, const VectorIndexMetadata& m) {
  j = nlohmann::json{{"name", m.config.name},
                     {"dimensions", m.config.dimensions},
                     {"metric", toString(m.config.metric)},
                     {"scalar", toString(m.config.scalar)},
                     {"numVectors", m.numVectors},
                     {"hasHnsw", m.hasHnsw},
                     {"hnswConnectivity", m.config.hnswConnectivity},
                     {"hnswExpansionAdd", m.config.hnswExpansionAdd},
                     {"hnswExpansionSearch", m.config.hnswExpansionSearch},
                     {"embeddingUrl", m.config.embeddingUrl},
                     {"embeddingModel", m.config.embeddingModel},
                     {"version", m.version}};
}

inline void from_json(const nlohmann::json& j, VectorIndexMetadata& m) {
  j.at("name").get_to(m.config.name);
  j.at("dimensions").get_to(m.config.dimensions);
  m.config.metric = vectorMetricFromString(j.at("metric").get<std::string>());
  m.config.scalar = vectorScalarFromString(j.at("scalar").get<std::string>());
  j.at("numVectors").get_to(m.numVectors);
  j.at("hasHnsw").get_to(m.hasHnsw);
  j.at("hnswConnectivity").get_to(m.config.hnswConnectivity);
  j.at("hnswExpansionAdd").get_to(m.config.hnswExpansionAdd);
  j.at("hnswExpansionSearch").get_to(m.config.hnswExpansionSearch);
  m.config.embeddingUrl = j.value("embeddingUrl", std::string{});
  m.config.embeddingModel = j.value("embeddingModel", std::string{});
  j.at("version").get_to(m.version);
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXFORMAT_H
