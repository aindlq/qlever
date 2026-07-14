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

// A `ScoredEntity` that ALSO carries the survivor's store ROW. Returned by the
// coarse pass (`searchExactCoarseWithRows`) so the SERVICE's fine rerank can
// score exactly these rows (`searchExactByRows`) WITHOUT re-deriving them from
// the entity ids -- the coarse pass already computed the rows, and recovering
// them via a second merge-join against the whole `.rowmap` was the fixed
// O(numVectors) cost that dominated the rerank.
struct ScoredRow {
  Id entity_;
  uint64_t row_;
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

// The TOP-ANCHORED Z-CUT (`vec:autoCut` coverage modes with the `cosine`/`csls`
// signal): the reranked candidates are kept when their signal score lies within
// a FRACTION `f(mode)` of the way DOWN from the best score to the noise floor
// -- `score >= s_max - f*(s_max - mu)`, equivalently `>= (1-f)*s_max + f*mu` --
// where `mu` is the noise-floor median (1.4826*MAD gives the spread `sigma`,
// still used by the `exact` gate). Anchoring the band to the TOP (not the
// floor) is what tames a SMOOTH cross-modal relevance gradient (text->image): a
// text vector is broadly close to a huge swath of the corpus, so a
// floor-anchored cut swept up thousands of mildly-related images; the
// top-anchored cut keeps the band right below the best. The `fraction` dial
// (vs. the earlier sigma-unit `Delta`) is query-INDEPENDENT: `f=0` keeps only
// the best, `f=1` keeps down to the floor, regardless of how many sigmas the
// curve happens to span. Still scale-INVARIANT (s_max and mu scale together).
// USER-TUNABLE knobs (also per-index overridable via the
// `QLEVER_VECTOR_SEARCH_ENDPOINTS` `zcut*` keys):
//   f(mode): how far down toward the floor to keep, a fraction in (0, 1].
//   precise ~0.3 (tight to the best), balanced ~0.6, broad ~0.85 (near the
//   floor) -- so precise's keep set is always a subset of balanced's, of
//   broad's (a larger f is a lower band, hence more).
inline constexpr float DEFAULT_ZCUT_FRACTION_PRECISE = 0.3f;
inline constexpr float DEFAULT_ZCUT_FRACTION_BALANCED = 0.6f;
inline constexpr float DEFAULT_ZCUT_FRACTION_BROAD = 0.85f;
// The NO-MATCH gate (only `exact` uses it): the answer is empty unless the
// BEST score itself stands `gateZ` spreads above the noise floor
// (`s_max >= mu_floor + gateZ*sigma`). This stays SIGMA-based (sigma is still
// estimated, just no longer the band unit). Robust even on a smooth curve -- a
// real match's best is far above the floor, a no-match's is not. precise/
// balanced/broad skip the gate (they always keep at least the single best).
inline constexpr float DEFAULT_ZCUT_GATE_Z = 3.0f;
// The background is the LOW `floorFraction` of the reranked window (sorted by
// signal): its median is the floor mu, and 1.4826 * MAD is the spread sigma
// (a std-equivalent robust to the standout head).
inline constexpr float DEFAULT_ZCUT_FLOOR_FRACTION = 0.5f;
inline constexpr float ZCUT_MAD_TO_SIGMA = 1.4826f;
// The adaptive rerank widens in `cslsRerankFloor`-sized batches until the
// window's MINIMUM score falls below the broadest cut's boundary
// `s_max - widenFraction*(s_max - mu)`, where `widenFraction = f_broad +
// WIDEN_MARGIN` is pushed just past the floor median (using the running-MIN mu)
// so the widen keeps progressing through a homogeneous top instead of stalling
// -- the cached set then already contains everything ANY mode could keep, and
// the widen stops. Because `widenFraction >= f_broad` and the running-min mu is
// <= the final mu, this widen boundary is always <= the broad KEEP band, so the
// cached set covers broad by construction. The cap still hard-bounds a genuine
// no-match (whose window never separates).
inline constexpr float DEFAULT_ZCUT_WIDEN_MARGIN = 0.25f;
// The softmax bar alpha per coverage mode (stricter = fewer standouts). These
// reproduce the former `vec:breadth` preset mapping (balanced == the old
// default), so a `cutSignal "softmax"` z-cut is bit-identical to today's
// softmax cut for the matching mode.
inline constexpr float DEFAULT_ZCUT_SOFTMAX_ALPHA_PRECISE = 4.0f;
inline constexpr float DEFAULT_ZCUT_SOFTMAX_ALPHA_BALANCED = 2.0f;
inline constexpr float DEFAULT_ZCUT_SOFTMAX_ALPHA_BROAD = 1.0f;

// The fully-resolved cut a `searchCsls*` call applies to the reranked
// candidate set. The classic fixed cut is `Mode::Threshold` (a bare float
// converts implicitly, so `searchCsls(query, 0.5f, ...)` keeps meaning "fixed
// tau 0.5"); the dynamic cuts are resolved from the query parameters, the
// per-index serving defaults, and the constants above by `resolveCslsCut`
// (`VectorSearch.h`). Fields not belonging to the active mode are ignored.
struct CslsCut {
  // Threshold: the fixed tau `csls >= tau`. Knee: the CSLS-gap cut (retained
  // for the `CslsCut`-level engine API; the `vec:autoCut` surface no longer
  // routes to it). Softmax: the softmax-standout cut. ZCut: the noise-floor
  // z-cut (the current `vec:autoCut` coverage modes, `cosine`/`csls` signal).
  enum class Mode { Threshold, Knee, Softmax, ZCut };
  // The signal the ZCut z-scores: the raw cosine SIMILARITY, or the CSLS value
  // `2*cos - r(q) - r(d)`. (`Softmax` always uses cosine; the others compute a
  // CSLS value.)
  enum class Signal { Cosine, Csls };
  // Implicit on purpose: a bare float IS the fixed-threshold cut.
  CslsCut(float threshold = 0.0f) : threshold_{threshold} {}

  Mode mode_ = Mode::Threshold;
  // Threshold: the tau of `csls >= tau`. Knee: the FLOOR -- candidates below
  // it never survive, and the two-layer rerank widening stops at it exactly
  // like a fixed tau. Unused by Softmax/ZCut.
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

  // ZCut (and the shared coverage-mode floor for Softmax):
  Signal signal_ = Signal::Cosine;
  // The top-anchored band FRACTION f(mode): keep candidates whose signal is
  // within `fraction_` of the way down from the best to the floor median
  // (`score >= s_max - fraction_*(s_max - mu)`). f in (0, 1].
  float fraction_ = DEFAULT_ZCUT_FRACTION_BALANCED;
  // The no-match gate (only when `keepAtLeastOne_` is false, i.e. `exact`):
  // keep nothing unless `s_max >= mu_floor + gateZ_*sigma` (still sigma-based).
  float gateZ_ = DEFAULT_ZCUT_GATE_Z;
  // Background = the low `floorFraction_` of the reranked window.
  float floorFraction_ = DEFAULT_ZCUT_FLOOR_FRACTION;
  // The adaptive-rerank widen depth: rerank until the window's min score falls
  // below `s_max - widenFraction_*(s_max - mu)` (see
  // `DEFAULT_ZCUT_WIDEN_MARGIN`).
  float widenFraction_ =
      DEFAULT_ZCUT_FRACTION_BROAD + DEFAULT_ZCUT_WIDEN_MARGIN;
  // Non-`exact` coverage modes keep at least one result (the single best when
  // nothing clears the band); `exact` keeps zero when the gate fails -- the
  // no-match answer. Defaults to false so a bare `CslsCut` (the direct engine
  // API) applies the gate; `resolveCslsCut` sets it per coverage mode.
  bool keepAtLeastOne_ = false;
  // Adaptive rerank cap: the widening never reranks more than this many
  // candidates (0 = the whole scored set). The initial batch is
  // `cslsRerankFloor`.
  size_t rerankCap_ = 0;

  // True for the dynamic cuts whose rerank depth is driven by the noise-floor
  // PLATEAU rather than a fixed threshold boundary.
  bool usesPlateauRerank() const { return mode_ == Mode::ZCut; }
};

// The MODE-INDEPENDENT product of a two-/single-layer CSLS search: the
// candidates reranked to the noise-floor plateau (or the rerank cap), in
// COARSE-RANK order, with everything a cut of ANY coverage mode / signal
// needs. Produced by `computeCslsReranked`, consumed by `applyCslsCut`; the
// engine caches it so switching `vec:autoCut`/`vec:cutSignal` on a repeat
// query re-applies the cut in O(reranked) with no rescan. Storing the fine
// cosine distance + the store row (for r(d) via the `.csls` sidecar) + r(q)
// is enough to derive the cosine, CSLS, and softmax signals.
struct CslsReranked {
  std::vector<uint64_t> entityBits_;  // survivor entity id bits, coarse order
  std::vector<uint64_t> storeRows_;   // its store row (r(d) sidecar lookup)
  std::vector<float> fineDist_;       // fine cosine DISTANCE (1 - cos_sim)
  float rq_ = 0.f;                    // r(q) over the reranked prefix
  size_t scored_ = 0;                 // number of candidates scored (= n)
  bool hasCsls_ = false;              // whether the index carries a `.csls`
  bool plateauFound_ = true;          // false => the rerank hit its cap
  size_t rerankDepth() const { return entityBits_.size(); }
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
  // Per-index serving defaults of the TOP-ANCHORED z-cut knobs (the `zcut*`
  // keys of `QLEVER_VECTOR_SEARCH_ENDPOINTS`): the per-mode band fractions
  // f(precise/balanced/broad), the exact no-match gate, and the floor-estimator
  // fraction. `nullopt` = the `DEFAULT_ZCUT_*` constant. The setters ignore
  // non-finite/out-of-(0,1] fractions and non-positive gate values.
  std::optional<float> zcutFractionDefault(
      int mode) const;  // 0=prec,1=bal,2=broad
  void setZcutFractionDefault(int mode, std::optional<float> fraction);
  std::optional<float> zcutGateZDefault() const;
  void setZcutGateZDefault(std::optional<float> gateZ);
  std::optional<float> zcutFloorFractionDefault() const;
  void setZcutFloorFractionDefault(std::optional<float> fraction);
  // The softmax temperature calibrated from the corpus at BUILD time (read from
  // the `.meta`; see `VectorIndexMetadata::calibratedSoftmaxT_`). Ranks below a
  // runtime-config default and a per-query override, above the constant
  // fallback (see `resolveCslsCut`). Absent for older/ingested/non-csls builds.
  std::optional<float> calibratedSoftmaxTemperature() const;

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
  // `bf16Kernel` (the `vec:bf16Kernel` performance dial) selects the exact
  // bf16-cosine kernel of the FINE layer -- a no-op on any other layer and
  // always result-identical to ~1e-6 across kernels.
  // `i8Kernel` (the `vec:i8Kernel` dial) is the i8-cosine sibling for the
  // layer being searched (the COARSE scan layer of a two-layer index, or the
  // single i8 layer): `Auto`/`Vnni` = the multi-row VNNI block engine,
  // `Punned` = the per-row engine. On a VNNI CPU both engines compute the
  // identical distances (one shared integer-dot + finalize), so the dial is
  // a pure performance A/B; without VNNI it is a no-op (punned metric).
  // `keepAll` (the SERVICE's FORM P annotate form, which must score and
  // return EVERY bound candidate): clamp `k` only to the live/candidate
  // bound, NOT to the hard `MAX_SEARCH_RESULTS` cap -- the result size is
  // then bounded by the caller's already-materialized candidate table, so
  // the cap's OOM-lever rationale does not apply. Same meaning on every
  // `searchExact*` overload below that takes it.
  std::vector<ScoredEntity> searchExact(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      bool keepAll = false, I8Kernel i8Kernel = I8Kernel::Auto) const;

  // The fine pass of the two-layer rerank over a set of survivors whose store
  // ROWS are already known (from `searchExactCoarseWithRows`): score exactly
  // those `rows` on the FINE layer and return the top-`k` ascending by
  // distance. This skips the candidate-id copy/sort and the O(numVectors)
  // `.rowmap` merge-join that `searchExact(query, k, <ids>)` pays to recover
  // rows the coarse pass had already computed. `rows` need not be sorted (the
  // top-k over a fixed set is order-independent for distinct distances);
  // duplicate rows are scored once each but that never changes the top-k.
  // `maxDistance`, if set, drops everything farther than it.
  std::vector<ScoredEntity> searchExactByRows(
      ql::span<const float> query, size_t k, ql::span<const ScoredRow> rows,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      Bf16Kernel bf16Kernel = Bf16Kernel::Auto, bool keepAll = false) const;

  // The same exact brute-force search on the COARSE scan matrix (identical to
  // `searchExact` on a single-layer index). This is the SERVICE's coarse
  // candidate pass of a two-layer index: cheap quantized bytes, results to be
  // re-scored on the fine layer.
  std::vector<ScoredEntity> searchExactCoarse(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, I8Kernel i8Kernel = I8Kernel::Auto) const;

  // Exactly `searchExactCoarse`, but each survivor ALSO carries its store row
  // (`ScoredRow`). This is the coarse half of the two-layer rerank: the
  // SERVICE feeds the returned rows straight to `searchExactByRows` for the
  // fine pass, so the row does not have to be re-derived from the entity id.
  // Results are ascending by (coarse) distance; at most `k` entries.
  std::vector<ScoredRow> searchExactCoarseWithRows(
      ql::span<const float> query, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, bool keepAll = false,
      I8Kernel i8Kernel = I8Kernel::Auto) const;

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
      size_t* numScored = nullptr, Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      bool keepAll = false, I8Kernel i8Kernel = I8Kernel::Auto) const;
  std::vector<ScoredEntity> searchExactCoarseByEntity(
      Id entity, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, I8Kernel i8Kernel = I8Kernel::Auto) const;

  // The by-entity coarse-with-rows / rerank-by-rows pair, mirroring
  // `searchExactCoarseWithRows` / `searchExactByRows` for a STORED entity as
  // the query point (the `vec:query <iri>` rerank flow).
  std::vector<ScoredRow> searchExactCoarseByEntityWithRows(
      Id entity, size_t k,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, bool keepAll = false,
      I8Kernel i8Kernel = I8Kernel::Auto) const;
  std::vector<ScoredEntity> searchExactByRowsByEntity(
      Id entity, size_t k, ql::span<const ScoredRow> rows,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      Bf16Kernel bf16Kernel = Bf16Kernel::Auto, bool keepAll = false) const;
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
  // `fullPrecision` (the `vec:fullPrecision` query flag) forces an exhaustive
  // sweep of the FINE layer with NO coarse preselection, even on a two-layer
  // index -- i.e. the single-layer code path -- so the search runs directly on
  // the highest-precision layer (e.g. bf16). A no-op on single-layer indices.
  std::vector<CslsScoredEntity> searchCsls(
      ql::span<const float> query, const CslsCut& cut, size_t neighbors,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      std::optional<float> maxDistance = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      size_t* numScored = nullptr, bool fullPrecision = false,
      Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      I8Kernel i8Kernel = I8Kernel::Auto) const;

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
      size_t* numScored = nullptr, bool fullPrecision = false,
      Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      I8Kernel i8Kernel = I8Kernel::Auto) const;

  // Stage (a) of the coverage-mode `vec:autoCut` (the `cosine`/`csls`/`softmax`
  // signals): rerank the candidates to the TOP-ANCHORED depth -- widen in
  // `cslsRerankFloor`-sized batches until the window's minimum cosine falls
  // below `s_max - widenFraction * (s_max - mu)` (so the reranked set already
  // contains everything any coverage mode could keep), or `rerankCap` (0 =
  // whole scored set) is hit -- and return the reranked set + r(q).
  // MODE-INDEPENDENT (the cut band `f(mode)`, gate, and signal are applied by
  // `applyCslsCut`), so the engine caches it and re-applies the cut for free on
  // a mode switch. `candidates`/`maxDistance` as `searchCsls`; `neighbors` is
  // the r(q) count.
  CslsReranked computeCslsReranked(
      ql::span<const float> query, size_t neighbors, float floorFraction,
      float widenFraction, size_t rerankCap,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      bool fullPrecision = false, Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      I8Kernel i8Kernel = I8Kernel::Auto) const;

  // The same with a STORED entity's vector as the query point.
  CslsReranked computeCslsRerankedByEntity(
      Id entity, size_t neighbors, float floorFraction, float widenFraction,
      size_t rerankCap,
      std::optional<ql::span<const Id>> candidates = std::nullopt,
      const CheckInterruptCallback& checkInterrupt = {},
      bool fullPrecision = false, Bf16Kernel bf16Kernel = Bf16Kernel::Auto,
      I8Kernel i8Kernel = I8Kernel::Auto) const;

  // Stage (b): apply a coverage-mode `cut` (ZCut over the cosine/CSLS signal,
  // or Softmax) to an already-`computeCslsReranked` set. O(reranked), no
  // rescan. `maxDistance`, if set, filters the OUTPUT. Non-`exact` modes keep
  // at least the single best; `exact` may keep zero (the no-match answer).
  std::vector<CslsScoredEntity> applyCslsCut(
      const CslsReranked& reranked, const CslsCut& cut,
      std::optional<float> maxDistance = std::nullopt) const;

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

// Whether every vector search should skip the quantized COARSE layer and score
// each candidate directly on the full-precision FINE layer (the rerank matrix,
// e.g. bf16) -- an exhaustive brute force with NO coarse preselection or rerank.
// Controlled once at process start by the environment variable
// `QLEVER_VECTOR_SEARCH_FULL_PRECISION` (a truthy value `1`/`true`/`on`/`yes`,
// case-insensitive). This makes a two-layer (binary + bf16) index behave exactly
// like a single-layer bf16 index, so the two can be A/B benchmarked on the SAME
// data with no rebuild. A no-op on single-layer indices (they already sweep
// their only layer) and on plain top-k as well as `vec:autoCut`/CSLS queries.
// Default OFF = the normal coarse-scan-then-rerank behaviour. Memoized.
bool vectorSearchFullPrecision();

namespace detail {
// TEST SEAM: mutable reference to the hard per-search result cap
// (`MAX_SEARCH_RESULTS`, default 100'000 -- see VectorIndex.cpp). Every
// `searchExact*`/`searchHnsw*` clamps its `k` to this cap unless the caller
// passes `keepAll` (the annotate form). Tests shrink it to exercise the cap
// (and the annotate form's exemption from it) on tiny fixtures without
// materializing 100k+ vectors; production code only ever READS it.
size_t& maxSearchResultsForTesting();
}  // namespace detail

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEX_H
