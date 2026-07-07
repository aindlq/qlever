// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H

#include <absl/functional/function_ref.h>

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

// Map a raw metric distance to the `vec:distance` result `Id`: a `NaN`
// distance (the "no live vector" sentinel) becomes UNDEF, any real distance
// becomes the double-encoded value. Defined ONCE here so the per-row
// `vec:distance` path and the sorted merge-walk fast path
// (`gatherSortedDistances`) share the exact same mapping and stay
// bit-identical.
Id distanceToValueId(float distance);

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
  // How eagerly to make the flat `.data` store resident in RAM. The default
  // relies on the page cache; the stronger levels trade memory for lower and
  // more predictable query latency. Each level gracefully degrades (with a
  // warning) if it cannot be satisfied -- e.g. the store does not fit in RAM,
  // `mlock` is denied, or the aligned copy cannot be allocated -- and it never
  // changes results, only paging/throughput.
  //   None        - rely on the OS page cache (madvise stays Random).
  //   Advise      - `madvise(MADV_WILLNEED)` prefault (advisory read-ahead).
  //   Lock        - additionally `mlock` the mapping so it is never paged out.
  //   AlignedCopy - copy the matrix into a 64-byte SIMD-aligned, huge-page
  //                 advised RAM buffer (also fixes alignment for legacy v4
  //                 files); holds a second full copy, so only for stores that
  //                 comfortably fit in RAM.
  enum class Residency { None, Advise, Lock, AlignedCopy };

 private:
  // Forward declaration so the public `DistanceComputer` below can hold an
  // opaque pointer to it; the definition lives in `VectorIndex.cpp`.
  struct Impl;

 public:
  VectorIndex();
  ~VectorIndex();
  VectorIndex(VectorIndex&&) noexcept;
  VectorIndex& operator=(VectorIndex&&) noexcept;
  VectorIndex(const VectorIndex&) = delete;
  VectorIndex& operator=(const VectorIndex&) = delete;

  // Open `<basename>.vec.<name>.*` read-only (everything is mmaped).
  // Throws if the files are missing or inconsistent. `residency` optionally
  // preloads/pins the flat store (see `Residency`); it is applied after a
  // successful open and never affects correctness.
  void open(const std::string& basename, const std::string& name,
            Residency residency = Residency::None);

  // Apply (or re-apply) a RAM-residency strategy to the already-opened flat
  // store. Idempotent and best-effort: gated on the store fitting in physical
  // memory (skipped with a warning otherwise) so it can never drive the
  // machine into OOM. Never changes search results.
  void makeResident(Residency residency);

  // The RAM-residency strategy most recently applied (best-effort) to the
  // flat store: the `open(..., residency)` argument (the "preload" serving
  // setting). `None` if nothing was requested, the store is empty, or the
  // fits-in-RAM gate skipped the request; degradation WITHIN a level (mlock
  // denied, aligned-copy allocation failure) still reports the requested
  // level, matching the warnings `makeResident` logs. Observational only
  // (logging and tests).
  Residency residency() const;

  // Map a `preload` configuration string ("advise", "lock", "aligned") to the
  // corresponding `Residency`; anything else (including "none") maps to
  // `Residency::None`. Used for the per-index `preload` serving setting from
  // the `QLEVER_VECTOR_SEARCH_ENDPOINTS` environment variable (see
  // `VectorIndexExtension.h`), which the load hook threads into `open()`.
  static Residency residencyFromString(const std::string& s);

  // Metadata accessors.
  const VectorIndexMetadata& metadata() const;
  // Override the IN-MEMORY embedding endpoint of this index (what query-time
  // `vec:embed` and the SERVICE's text/image query points use): a `nullopt`
  // field keeps its current value. This never rewrites the on-disk `.meta`,
  // so it is a pure runtime override that has to be reapplied after every
  // `open()` -- the load hook does exactly that from the
  // `QLEVER_VECTOR_SEARCH_ENDPOINTS` environment variable (see
  // `VectorIndexExtension.h`).
  void setEmbeddingEndpoint(std::optional<std::string> url,
                            std::optional<std::string> model);
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

  // A reusable per-query distance functor. It encodes the query point ONCE
  // (into the index's storage scalar) and owns those bytes, then computes the
  // metric distance from that point to any entity's stored vector via a single
  // `.rowmap` lookup + one SIMD kernel call. `operator()` returns `NaN` for an
  // entity that has no live vector in this index. This is the primitive behind
  // the `vec:distance` SPARQL expression: BIND it over a bound `?entity`, then
  // ORDER BY + LIMIT to get a filtered top-k search using QLever's own
  // machinery. Cheap to copy; only valid while the `VectorIndex` it was made
  // from is alive (it borrows the index's metric/mapping).
  class DistanceComputer {
   public:
    // The distance from the query point to `entity`'s stored vector, or `NaN`
    // if `entity` has no (live) vector in this index.
    float operator()(Id entity) const;

    // The distance from the query point to a raw f32 `vector` (encoded into
    // the storage scalar first), or `NaN` if `vector`'s dimension does not
    // match the index.
    float operator()(ql::span<const float> vector) const;

    // The distance from the query point to an ALREADY-KNOWN row of the flat
    // store, with NO rowmap lookup. This is the primitive of the sorted
    // merge-walk fast path (`gatherSortedDistances`), which resolves the row
    // once while walking the id-sorted rowmap. For a row that came from
    // `rowOf(entity)` this returns exactly `(*this)(entity)`. Precondition:
    // `row < numVectors()` (the walk only ever passes a valid rowmap row).
    float atRow(size_t row) const;

   private:
    friend class VectorIndex;
    DistanceComputer(const Impl* impl, std::vector<char> queryBytes)
        : impl_{impl}, queryBytes_{std::move(queryBytes)} {}
    const Impl* impl_;
    // The query point, already encoded into the storage scalar (length
    // `rowBytes`), owned so the functor does not depend on the caller's buffer.
    std::vector<char> queryBytes_;
  };

  // Make a `DistanceComputer` for an explicit f32 `query` (encoded once into
  // the storage scalar). Throws if `query`'s dimension does not match the
  // index.
  DistanceComputer makeDistanceComputer(ql::span<const float> query) const;

  // Make a `DistanceComputer` whose query point is the stored vector of
  // `entity` (used directly, no f32 round trip). Returns `nullopt` if `entity`
  // has no (live) vector in this index.
  std::optional<DistanceComputer> makeDistanceComputerByEntity(Id entity) const;

  // One-shot pairwise distances in the index's metric (used by the generalized
  // `vec:distance` expression when neither side is a fixed query point, so no
  // reusable `DistanceComputer` pays off). Entity vectors are used as stored
  // (no f32 round trip); raw f32 vectors are encoded into the storage scalar.
  // Return `NaN` when an entity has no (live) vector in this index or a raw
  // vector's dimension does not match the index.
  float distance(Id a, Id b) const;
  float distance(Id entity, ql::span<const float> vector) const;
  float distance(ql::span<const float> a, ql::span<const float> b) const;

  // Sorted-input fast path for `vec:distance`: compute the distance from
  // `computer`'s query point to every entity in `ascendingEntities`, which
  // MUST be sorted ascending by `Id` bits (the caller only takes this path
  // when the entity column is the input's primary sort key, so it matches the
  // physical id order of the store). Instead of a rowmap binary search per
  // row, MERGE-WALK the column against the id-sorted rowmap in one linear pass
  // (one `lower_bound` per parallel chunk, then a forward cursor), and reuse
  // the previous result for consecutive duplicate ids so a repeated entity
  // costs one SIMD distance, not N. Writes `out[i]` for every `i`:
  //  * live member    -> `distanceToValueId(computer.atRow(row))`, where `row`
  //                      is the SAME row `rowOf` would find, so the result is
  //                      bit-identical to the per-row path's
  //                      `distanceToValueId((*computer)(entity))`;
  //  * everything else -> `onMiss(i)` (a non-member id, UNDEF, or -- for the
  //                      `vec:distance` expression -- a per-row float-list
  //                      literal; `onMiss` is only ever invoked for these
  //                      NON-member rows, so members never pay for it).
  // Runs in parallel over contiguous chunks capped at `numThreads` workers
  // (<= 1, or a non-OpenMP build => serial); `isCancelled` is polled once per
  // chunk (non-throwing -- a cancelled chunk leaves its caller-prefilled
  // `out[i]`, and the caller raises the actual cancellation afterwards).
  // `out.size()` must equal `ascendingEntities.size()`. `onMiss` and
  // `isCancelled` may be called concurrently from the workers, so they must be
  // thread-safe const reads (the expression's are).
  void gatherSortedDistances(ql::span<const Id> ascendingEntities,
                             const DistanceComputer& computer,
                             absl::FunctionRef<Id(size_t)> onMiss,
                             absl::FunctionRef<bool()> isCancelled,
                             int numThreads, ql::span<Id> out) const;

 private:
  std::unique_ptr<Impl> impl_;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
