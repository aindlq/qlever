// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "backports/span.h"
#include "global/Id.h"
#include "services/vectorSearch/VectorIndexFormat.h"

namespace qlever::vector {

// One result of a similarity search: an entity and its distance to the query
// (smaller = more similar, in the index's metric). The entity is identified by
// its full `Id` (`ValueId`) -- the same value a join variable binds to -- so
// the index works uniformly for vocabulary IRIs, encoded IRIs, blank nodes,
// etc.
struct ScoredEntity {
  Id entity_;
  float distance_;
};

// A callback that long-running searches invoke periodically so the caller can
// throw on query cancellation/timeout (see `Operation::checkCancellation`).
// An empty function disables the checks.
using CheckInterruptCallback = std::function<void()>;

// Read-only, memory-mapped accessor for a single on-disk vector index (see
// `VectorIndexFormat.h` for the file layout). It owns the mmaped flat float
// store, the entity mappings, and, if present, the memory-mapped usearch HNSW
// graph. All search methods are const and thread-safe for concurrent reads;
// concurrent HNSW searches beyond the reserved context pool briefly queue
// instead of failing.
//
// The exact and approximate searches use the *same* metric implementation
// (usearch's `metric_punned_t`), so their distances are directly comparable.
// Tombstoned rows (entities that disappeared from the knowledge graph after a
// remap) are skipped by all searches.
class VectorIndex {
 public:
  VectorIndex();
  ~VectorIndex();
  VectorIndex(VectorIndex&&) noexcept;
  VectorIndex& operator=(VectorIndex&&) noexcept;
  VectorIndex(const VectorIndex&) = delete;
  VectorIndex& operator=(const VectorIndex&) = delete;

  // Open `<basename>.vec.<name>.*` read-only (everything is mmaped).
  // Throws if the files are missing or inconsistent.
  void open(const std::string& basename, const std::string& name);

  // Metadata accessors.
  const VectorIndexMetadata& metadata() const;
  size_t dimensions() const;
  // Number of rows in the flat store (including tombstoned rows).
  size_t numVectors() const;
  // Number of rows whose entity exists in the current knowledge graph.
  size_t numLiveVectors() const;
  VectorMetric metric() const;
  bool hasHnsw() const;

  // True iff this index stores a (live) vector for `entity`.
  bool hasVector(Id entity) const;

  // The stored vector of `entity` decoded to f32, or `nullopt` if this index
  // has none for it. (The store may hold f16/i8; searching BY an entity does
  // not decode -- see the `...ByEntity` methods.)
  std::optional<std::vector<float>> getVector(Id entity) const;

  // Exact brute-force top-`k` nearest neighbours of `query`.
  //  - If `candidates` is `nullopt`, searches over ALL (live) entities.
  //  - Otherwise searches only the given candidate entities (the optimisation
  //    used when a join's search side is already small). Candidates without a
  //    vector in this index are skipped; an EMPTY candidate list yields an
  //    empty result (it does NOT fall back to the whole index).
  // Results are ascending by distance; at most `k` entries. `maxDistance`, if
  // set, drops everything farther than it.
  std::vector<ScoredEntity> searchExact(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;

  // Approximate top-`k` via the HNSW graph over the whole index. Requires
  // `hasHnsw()`. Results are ascending by distance. `k` is clamped to the
  // number of live vectors. `checkInterrupt`, if set, is polled while waiting
  // for a search slot so the search can be cancelled under load.
  std::vector<ScoredEntity> searchHnsw(
      ql::span<const float> query, size_t k,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;

  // The same two searches with a STORED entity's vector as the query point
  // (used by the join form and `vec:query <iri>`). The stored bytes are used
  // directly -- no decode/re-encode round trip through f32. An entity without
  // a (live) vector yields an empty result.
  std::vector<ScoredEntity> searchExactByEntity(
      Id entity, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;
  std::vector<ScoredEntity> searchHnswByEntity(
      Id entity, size_t k, std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
