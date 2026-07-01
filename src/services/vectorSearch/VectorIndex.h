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

// Read-only, memory-mapped accessor for a single on-disk vector index. It owns
// the mmaped flat float store (`.keys`/`.data`) and, if present, the
// memory-mapped usearch HNSW index (`.hnsw`). All search methods are const and
// thread-safe for concurrent reads (usearch search is thread-safe).
//
// The exact and approximate searches use the *same* metric implementation
// (usearch's `metric_punned_t`), so their distances are directly comparable.
class VectorIndex {
 public:
  VectorIndex();
  ~VectorIndex();
  VectorIndex(VectorIndex&&) noexcept;
  VectorIndex& operator=(VectorIndex&&) noexcept;
  VectorIndex(const VectorIndex&) = delete;
  VectorIndex& operator=(const VectorIndex&) = delete;

  // Open `<basename>.vec.<name>.*` read-only (vectors and HNSW are mmaped).
  // Throws if the metadata/keys/data files are missing or inconsistent.
  void open(const std::string& basename, const std::string& name);

  // Metadata accessors.
  const VectorIndexMetadata& metadata() const;
  size_t dimensions() const;
  size_t numVectors() const;
  VectorMetric metric() const;
  bool hasHnsw() const;

  // The stored vector of `entity`, or `nullopt` if this index has none for it.
  // The span points into the mmaped store and is valid for this object's
  // lifetime.
  std::optional<ql::span<const float>> getVector(Id entity) const;

  // Exact brute-force top-`k` nearest neighbours of `query`.
  //  - If `candidates` is `nullopt`, searches over ALL entities in the index.
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

  // Approximate top-`k` via the HNSW index over the whole index. Requires
  // `hasHnsw()`. Results are ascending by distance.
  std::vector<ScoredEntity> searchHnsw(
      ql::span<const float> query, size_t k,
      std::optional<float> maxDistance = std::nullopt) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
