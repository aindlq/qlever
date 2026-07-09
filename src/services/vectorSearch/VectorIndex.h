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

// One survivor of a CSLS-filtered search (`searchCsls*`): the entity, its
// COSINE DISTANCE to the query (what `vec:bindScore` binds and what results
// sort by -- CSLS is the cut, not the score), and its CSLS value
// `2 * cos_sim(q, d) - r(q) - r(d)` (>= the caller's threshold/floor; what
// `vec:bindCsls` binds). NaN for the softmax autoCut mode, which does not
// define a CSLS value (`vec:bindCsls` is rejected with it at parse time).
struct CslsScoredEntity {
  Id entity_;
  float distance_;
  float csls_;
};

// A callback that long-running searches invoke periodically so the caller can
// throw on query cancellation/timeout (see `Operation::checkCancellation`).
// An empty function disables the checks.
using CheckInterruptCallback = std::function<void()>;

// Default fine-rerank batch size of the two-layer CSLS cut (see
// `VectorIndex::cslsRerankFloor()`): the coarse scan ranks ALL candidates,
// only the coarse-best this-many are reranked on the fine layer (widened
// automatically while the cut reaches the coarse boundary). Large enough that
// the r(q) neighbourhood (typically ~10 fine cosines) is effectively exact.
inline constexpr size_t DEFAULT_CSLS_RERANK_FLOOR = 10'000;

// Constant fallback defaults of the DYNAMIC `vec:autoCut` cuts (each
// overridable per query and per index at serving time; see `CslsCut` below
// and `resolveCslsCut` in `VectorSearch.h`). These are query-time defaults; a
// later batch may data-calibrate them from the build-time self-kNN.
//  * knee ("csls"): candidates below the FLOOR never survive; the knee only
//    counts when its gap exceeds SIGNIFICANCE x the median gap of the
//    inspected head (else the cut falls back to "everything >= floor"), and
//    the knee search inspects at most the top MAX_KEEP survivors by CSLS.
//  * softmax: the top `SOFTMAX_N_FACTOR * cslsNeighbors` fine cosines enter
//    `softmax(cos / TEMPERATURE)`; survivors need `p_i >= ALPHA / N` (at
//    least ALPHA x the uniform 1/N).
//  * BREADTH 0.5 is the neutral dial position (the defaults above).
inline constexpr float DEFAULT_CSLS_FLOOR = 0.0f;
inline constexpr float DEFAULT_CSLS_KNEE_SIGNIFICANCE = 3.0f;
inline constexpr size_t DEFAULT_CSLS_KNEE_MAX_KEEP = 1'000;
inline constexpr float DEFAULT_CSLS_SOFTMAX_TEMPERATURE = 0.1f;
inline constexpr size_t DEFAULT_CSLS_SOFTMAX_N_FACTOR = 5;
inline constexpr float DEFAULT_CSLS_SOFTMAX_ALPHA = 2.0f;
inline constexpr float DEFAULT_CSLS_AUTOCUT_BREADTH = 0.5f;

// The fully-resolved cut a `searchCsls*` call applies to the reranked
// candidate set. The classic fixed cut is `Mode::Threshold` (a bare float
// converts implicitly, so `searchCsls(query, 0.5f, ...)` keeps meaning "fixed
// tau 0.5"); the dynamic cuts are resolved from the query parameters, the
// per-index serving defaults, and the constants above by `resolveCslsCut`
// (`VectorSearch.h`). Fields not belonging to the active mode are ignored.
struct CslsCut {
  enum class Mode { Threshold, Knee, Softmax };
  // Implicit on purpose: a bare float IS the fixed-threshold cut.
  CslsCut(float threshold = 0.0f) : threshold_{threshold} {}

  Mode mode_ = Mode::Threshold;
  // Threshold: the tau of `csls >= tau`. Knee: the FLOOR -- candidates below
  // it never survive, and the two-layer rerank widening stops at it exactly
  // like a fixed tau. Unused by Softmax.
  float threshold_;
  // Knee: the knee is taken only if its gap > `significanceFactor_` x the
  // median gap of the inspected head (else: keep everything >= the floor),
  // and the knee search inspects at most the top-`maxKeep_` survivors by
  // CSLS descending (tail noise never moves the knee).
  float significanceFactor_ = DEFAULT_CSLS_KNEE_SIGNIFICANCE;
  size_t maxKeep_ = DEFAULT_CSLS_KNEE_MAX_KEEP;
  // Softmax: `p_i = softmax(cos_i / temperature_)` over the top-`softmaxN_`
  // fine cosines (capped at the reranked count); keep `p_i >= alpha_ / N`
  // with N the EFFECTIVE (capped) count. `softmaxN_` must be >= 1 when
  // `mode_ == Softmax` (the resolver guarantees it).
  size_t softmaxN_ = 0;
  float temperature_ = DEFAULT_CSLS_SOFTMAX_TEMPERATURE;
  float alpha_ = DEFAULT_CSLS_SOFTMAX_ALPHA;
};

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
//
// TWO-LAYER indices (`hasRerankLayer()`): the index holds a coarse SCAN
// matrix (`.data`, e.g. i8 -- what the HNSW graph and the explicitly-named
// `...Coarse` scans read) plus a fine RERANK matrix (`.rerank.data`, e.g.
// bf16). All the exact primitives below -- `searchExact*`, the
// `DistanceComputer`, the `distance` overloads, `getVector` -- read the FINE
// layer, so `vec:distance` stays the exact baseline; only `searchHnsw*` and
// `searchExactCoarse*` (the SERVICE's coarse candidate pass) read the scan
// layer. On a single-layer index the two coincide.
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
  // preloads/pins the flat SCAN store (see `Residency`); `rerankResidency`
  // does the same, INDEPENDENTLY, for the fine rerank store of a two-layer
  // index (ignored when there is none) -- so an operator can e.g. mlock the
  // small i8 scan matrix and leave the bf16 rerank matrix demand-paged. Both
  // are applied after a successful open and never affect correctness.
  void open(const std::string& basename, const std::string& name,
            Residency residency = Residency::None,
            Residency rerankResidency = Residency::None);

  // Apply (or re-apply) a RAM-residency strategy to the already-opened flat
  // SCAN store. Idempotent and best-effort: gated on the store fitting in
  // physical memory (skipped with a warning otherwise) so it can never drive
  // the machine into OOM. Never changes search results. (The rerank layer's
  // residency is load-time only: the `rerankResidency` argument of `open`.)
  void makeResident(Residency residency);

  // The RAM-residency strategy most recently applied (best-effort) to the
  // flat SCAN store: the `open(..., residency)` argument (the "preload"
  // serving setting). `None` if nothing was requested, the store is empty, or
  // the fits-in-RAM gate skipped the request; degradation WITHIN a level
  // (mlock denied, aligned-copy allocation failure) still reports the
  // requested level, matching the warnings `makeResident` logs. Observational
  // only (logging and tests).
  Residency residency() const;

  // The residency most recently applied to the fine RERANK store (the
  // "preloadRerank" serving setting). Always `None` on a single-layer index.
  Residency rerankResidency() const;

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
  // True iff this is a two-layer index (a fine `.rerank.data` matrix exists
  // next to the coarse scan `.data`; see the class comment).
  bool hasRerankLayer() const;

  // True iff this index was built with `csls: true` (a row-aligned `.csls`
  // hubness sidecar exists; a `.meta` without the field loads as
  // `hasCsls() == false`, so old indexes are unaffected).
  bool hasCsls() const;
  // The neighbour count k of the CSLS r-terms (both the persisted r(d) and
  // the query-time default for r(q)). Only meaningful with `hasCsls()`.
  size_t cslsNeighbors() const;
  // The persisted hubness r(d) of `row` (mean cosine similarity to its
  // `cslsNeighbors()` nearest corpus neighbours, self-excluded).
  // Preconditions: `hasCsls()` and `row < numVectors()`.
  float cslsRForRow(size_t row) const;
  // The same by entity (one `.rowmap` lookup); `nullopt` for an entity
  // without a (live) vector. Precondition: `hasCsls()`.
  std::optional<float> cslsRForEntity(Id entity) const;
  // Fine-rerank batch size M of the two-layer CSLS cut (`searchCsls*` on a
  // `hasRerankLayer()` index coarse-scans all candidates and reranks only the
  // coarse-best M on the fine layer, widening by M while the cut reaches the
  // coarse boundary). Defaults to `DEFAULT_CSLS_RERANK_FLOOR`; the setter is
  // a pure IN-MEMORY serving override (never persisted, reapplied by the load
  // hook from the `cslsRerankFloor` key of `QLEVER_VECTOR_SEARCH_ENDPOINTS`),
  // clamped to >= 1. Irrelevant on a single-layer index.
  size_t cslsRerankFloor() const;
  void setCslsRerankFloor(size_t floor);

  // Per-index serving DEFAULTS of the dynamic `vec:autoCut` cuts, consulted
  // by `resolveCslsCut` when the query does not override them; `nullopt`
  // falls through to the `DEFAULT_CSLS_*` constants. Like `cslsRerankFloor`
  // these are pure IN-MEMORY serving settings: never persisted, reapplied by
  // the load hook from the `cslsFloor`/`softmaxTemperature`/`softmaxN`/
  // `breadth` keys of `QLEVER_VECTOR_SEARCH_ENDPOINTS`. The setters clamp
  // defensively (the env parser already validates): the floor must be finite
  // (else ignored), the temperature positive and finite (else ignored),
  // `softmaxN` >= 1, `breadth` clamped into [0, 1].
  std::optional<float> cslsFloorDefault() const;
  void setCslsFloorDefault(std::optional<float> floor);
  std::optional<float> softmaxTemperatureDefault() const;
  void setSoftmaxTemperatureDefault(std::optional<float> temperature);
  std::optional<size_t> softmaxNDefault() const;
  void setSoftmaxNDefault(std::optional<size_t> n);
  std::optional<float> breadthDefault() const;
  void setBreadthDefault(std::optional<float> breadth);

  // True iff this index stores a (live) vector for `entity`.
  bool hasVector(Id entity) const;

  // Write the entity `Id`s of all LIVE members (entities that have a vector in
  // this index) into `out`, in ascending `ValueId` order. That is exactly the
  // physical order of the id-sorted `.rowmap` (validated strictly ascending by
  // `idBits_` at open): the store keeps entities in id order and tombstones are
  // already excluded. For the persistent ids stored here (never local-vocab)
  // the bit order equals QLever's `Id` comparison order, so the emitted column
  // is sorted the way a merge join expects. `out.size()` MUST equal
  // `numLiveVectors()`. This is the primitive behind the `vec:hasMember`
  // membership scan: emit the members as one already-sorted single column and
  // let the planner merge-join it -- no vectors are materialised.
  void memberEntities(ql::span<Id> out) const;

  // The stored vector of `entity` decoded to f32 (from the FINE layer of a
  // two-layer index), or `nullopt` if this index has none for it. (The store
  // may hold f16/i8; searching BY an entity does not decode -- see the
  // `...ByEntity` methods.)
  std::optional<std::vector<float>> getVector(Id entity) const;

  // Exact brute-force top-`k` nearest neighbours of `query`, scored on the
  // FINE layer (the rerank matrix when present, else the single store) -- the
  // exact baseline that `vec:distance` and the SERVICE's rerank pass share.
  //  - If `candidates` is `nullopt`, searches over ALL (live) entities.
  //  - Otherwise searches only the given candidate entities (the optimisation
  //    used when a join's search side is already small, and the SERVICE's
  //    rerank pass over the coarse top-`rerankK`). Candidates without a
  //    vector in this index are skipped; an EMPTY candidate list yields an
  //    empty result (it does NOT fall back to the whole index).
  // Results are ascending by distance; at most `k` entries. `maxDistance`, if
  // set, drops everything farther than it.
  std::vector<ScoredEntity> searchExact(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;

  // The same exact brute-force search on the COARSE scan matrix (identical to
  // `searchExact` on a single-layer index). This is the SERVICE's coarse
  // candidate pass of a two-layer index: cheap quantized bytes, results to be
  // re-scored on the fine layer.
  std::vector<ScoredEntity> searchExactCoarse(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;

  // Approximate top-`k` via the HNSW graph over the whole index. Requires
  // `hasHnsw()`. Results are ascending by distance. `k` is clamped to the
  // number of live vectors. `checkInterrupt`, if set, is polled while waiting
  // for a search slot so the search can be cancelled under load. NOTE: the
  // graph reads the COARSE scan matrix, so on a two-layer index the returned
  // distances are coarse-space (the SERVICE reranks them on the fine layer).
  std::vector<ScoredEntity> searchHnsw(
      ql::span<const float> query, size_t k,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;

  // The same searches with a STORED entity's vector as the query point (used
  // by the join form and `vec:query <iri>`). The stored bytes of the layer
  // being searched are used directly -- no decode/re-encode round trip
  // through f32. An entity without a (live) vector yields an empty result.
  std::vector<ScoredEntity> searchExactByEntity(
      Id entity, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;
  std::vector<ScoredEntity> searchExactCoarseByEntity(
      Id entity, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;
  std::vector<ScoredEntity> searchHnswByEntity(
      Id entity, size_t k, std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {}) const;

  // CSLS-filtered search (`vec:cslsThreshold` / `vec:autoCut`); requires
  // `hasCsls()` and the cosine metric (both guaranteed at build time;
  // violated preconditions throw). Every candidate is scored (the cut is a
  // decision over the whole scoring set, so there is no HNSW shortcut):
  //  1. compute the cosine distance from the query to EVERY candidate (all
  //     live vectors, or -- like `searchExact` -- only the `candidates` set).
  //     On a SINGLE-LAYER index this is one full fine sweep; on a TWO-LAYER
  //     index the full sweep runs on the cheap COARSE scan matrix and only
  //     the coarse-best `cslsRerankFloor()` candidates get a FINE distance,
  //     widened batch-by-batch while the cut still reaches the coarse
  //     boundary (so everything the cut could keep is reranked);
  //  2. r(q) = mean cosine similarity (`1 - distance`) of the query to its
  //     top-`neighbors` nearest candidates (on a two-layer index: nearest
  //     RERANKED candidates -- the only approximation of the coarse+rerank
  //     path), EXCLUDING one exact self-match (distance ~ 0 -- the query
  //     entity itself when it is a candidate);
  //  3. apply `cut` to the reranked set:
  //     * Threshold: keep candidate `d` iff
  //       `CSLS = 2 * cos_sim(q, d) - r(q) - r(d) >= cut.threshold_`;
  //     * Knee: the same with `cut.threshold_` as the FLOOR, then cut the
  //       survivors at a SIGNIFICANT largest CSLS gap (see `CslsCut`); the
  //       widening bound is the floor, exactly like a fixed tau;
  //     * Softmax: keep the standouts of `softmax(cos / T)` over the
  //       top-`softmaxN` fine cosines (r(q)/r(d) unused; `csls_` of the
  //       returned survivors is NaN). No widening -- the coarse layer ranks
  //       the very top well (the same approximation r(q) already makes), so
  //       one batch (>= `softmaxN`) suffices.
  //     In every mode `distance <= maxDistance` (if set) additionally
  //     filters the OUTPUT; r(q) is computed BEFORE the maxDistance filter
  //     -- it describes the retrieval geometry. The cosine is always the
  //     FINE-layer distance.
  // Returns ALL survivors ascending by cosine DISTANCE (the cut selects, the
  // cosine distance stays the score); the caller applies any top-k cap.
  // `numScored`, if set, receives the number of candidates that were scored
  // (the live set, or the members among `candidates`).
  std::vector<CslsScoredEntity> searchCsls(
      ql::span<const float> query, const CslsCut& cut, size_t neighbors,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;

  // The same with a STORED entity's vector as the query point (its stored
  // bytes of the layer being swept are used directly -- the fine row for the
  // fine distances, the scan row for a two-layer coarse sweep); the entity's
  // own row is the excluded self-match. An entity without a (live) vector
  // yields an empty result.
  std::vector<CslsScoredEntity> searchCslsByEntity(
      Id entity, const CslsCut& cut, size_t neighbors,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr) const;

  // A reusable per-query distance functor. It encodes the query point ONCE
  // (into the FINE layer's storage scalar -- the rerank matrix when present,
  // so `vec:distance` is always the exact baseline) and owns those bytes,
  // then computes the metric distance from that point to any entity's stored
  // vector via a single `.rowmap` lookup + one SIMD kernel call. `operator()`
  // returns `NaN` for an entity that has no live vector in this index. This
  // is the primitive behind the `vec:distance` SPARQL expression: BIND it
  // over a bound `?entity`, then ORDER BY + LIMIT to get a filtered top-k
  // search using QLever's own machinery. Cheap to copy; only valid while the
  // `VectorIndex` it was made from is alive (it borrows the index's
  // metric/mapping).
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

// The number of PHYSICAL cores on the machine. Hyperthread siblings share
// their core's execution ports and add no memory bandwidth, so for the
// memory-bandwidth-bound SIMD scans the right worker count is the physical
// core count, not the logical (hyperthread) one. Parsed from `/proc/cpuinfo`
// (unique `(physical id, core id)` pairs); falls back to
// `std::thread::hardware_concurrency()` (which may count hyperthreads) where
// that file is unavailable or lacks those fields (some containers/ARM).
// Memoized -- the topology never changes at runtime. NOTE: inside a
// cpu-limited container this still reflects the HOST's physical cores;
// `OMP_NUM_THREADS` remains the override for cgroup limits.
unsigned physicalCoreCount();

// Worker cap for the parallel brute-force distance scans (the top-k sweep and
// the `vec:distance` per-row loop). QLever answers many queries concurrently
// on a thread pool already sized to the hardware, so letting a single scan
// grab every logical CPU would oversubscribe when several queries run at
// once. Cap at the PHYSICAL core count (see `physicalCoreCount()`; the scans
// are memory-bandwidth bound, so there is nothing to gain past it) AND
// OpenMP's configured maximum, so `OMP_NUM_THREADS` can still lower it (e.g.
// for cgroup cpu limits). If the environment variable
// `QLEVER_VECTOR_SEARCH_THREADS` holds a positive integer, it replaces the
// physical-core count (still capped by OpenMP's maximum); anything else
// leaves the default. Search-side only -- the index-build thread counts are
// chosen separately.
int vectorSearchThreadCap();

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
