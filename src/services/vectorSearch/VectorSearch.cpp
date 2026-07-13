// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSearch.h"

#include <absl/base/casts.h>
#include <absl/strings/str_cat.h>

#include <cmath>

#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorQueryPoint.h"
#include "util/Cache.h"
#include "util/ConcurrentCache.h"
#include "util/HashMap.h"
#include "util/MemorySize/MemorySize.h"
#include "util/Timer.h"

namespace {
// Append a string to a cache key unambiguously (length-prefixed, so crafted
// values cannot collide with other parts of the key).
void appendToKey(std::string* key, std::string_view field,
                 std::string_view value) {
  absl::StrAppend(key, " ", field, "=", value.size(), ":", value);
}
}  // namespace

namespace qlever::vector {

// ____________________________________________________________________________
void appendQueryPointToCacheKey(std::string* key,
                                const VectorSearchConfiguration& config) {
  if (config.queryEntityIri_.has_value()) {
    appendToKey(key, "entity", config.queryEntityIri_.value());
  }
  if (config.queryVector_.has_value()) {
    absl::StrAppend(key, " vec=[");
    for (float f : config.queryVector_.value()) {
      // Bit-exact: `absl::StrCat(float)` would round to six significant
      // digits, making distinct query vectors collide on one key.
      absl::StrAppend(key, absl::Hex(absl::bit_cast<uint32_t>(f)), ",");
    }
    absl::StrAppend(key, "]");
  }
  if (config.queryText_.has_value()) {
    appendToKey(key, "text", config.queryText_.value());
  }
  if (config.queryImage_.has_value()) {
    absl::StrAppend(
        key, " imageKind=", static_cast<int>(config.queryImage_.value().kind_));
    appendToKey(key, "image", config.queryImage_.value().value_);
  }
}

// ____________________________________________________________________________
void appendCslsCutToCacheKey(std::string* key,
                             const VectorSearchConfiguration& config) {
  if (!config.hasCslsCut()) {
    return;
  }
  if (config.cslsThreshold_.has_value()) {
    absl::StrAppend(
        key, " cslsThreshold=",
        absl::Hex(absl::bit_cast<uint32_t>(config.cslsThreshold_.value())));
  } else {
    absl::StrAppend(key, " autoCut=", static_cast<int>(config.autoCut_.value()),
                    " cutSignal=", static_cast<int>(config.cutSignal_));
    if (config.softmaxTemperature_.has_value()) {
      absl::StrAppend(key, " softmaxT=",
                      absl::Hex(absl::bit_cast<uint32_t>(
                          config.softmaxTemperature_.value())));
    }
    if (config.softmaxN_.has_value()) {
      absl::StrAppend(key, " softmaxN=", config.softmaxN_.value());
    }
  }
  absl::StrAppend(key, " csls=", config.cslsVariable_.has_value(),
                  " cslsKCap=", config.cslsKCap_.value_or(0));
  if (config.cslsNeighbors_.has_value()) {
    absl::StrAppend(key, " cslsNeighbors=", config.cslsNeighbors_.value());
  }
}

// ____________________________________________________________________________
void validateCslsIsAvailable(const VectorSearchConfiguration& config,
                             const VectorIndex& vidx) {
  // The cosine z-cut and the softmax cut are CSLS-INDEPENDENT (they never
  // consult r(d)/r(q)), so they need only a cosine-metric index. The fixed
  // `cslsThreshold` and the `csls`-signal z-cut use r(d), so they require the
  // `.csls` sidecar.
  if (!config.cutNeedsCsls()) {
    if (vidx.metric() != VectorMetric::Cosine) {
      throw std::runtime_error{absl::StrCat(
          "`vec:autoCut` (the cosine/softmax cut) needs a cosine-metric vector "
          "index, but vector index '",
          config.indexName_, "' uses a non-cosine metric.")};
    }
    return;
  }
  if (!vidx.hasCsls()) {
    throw std::runtime_error{absl::StrCat(
        "`vec:",
        config.cslsThreshold_.has_value() ? "cslsThreshold"
                                          : "cutSignal \"csls\"",
        "` requires an index built with `\"csls\": true`, but vector index '",
        config.indexName_, "' was not built with csls:true.")};
  }
}

namespace {
// The band-fraction index, softmax bar, and no-match behaviour of each coverage
// mode.
struct CoverageParams {
  int fractionMode_;  // 0=precise,1=balanced,2=broad -- indexes the per-mode f
  float softmaxAlpha_;
  bool keepAtLeastOne_;
};
CoverageParams coverageParams(VectorSearchConfiguration::CoverageMode mode) {
  using M = VectorSearchConfiguration::CoverageMode;
  switch (mode) {
    case M::Precise:
      return {0, DEFAULT_ZCUT_SOFTMAX_ALPHA_PRECISE, true};
    case M::Broad:
      return {2, DEFAULT_ZCUT_SOFTMAX_ALPHA_BROAD, true};
    case M::Exact:
      // Exact = precise's band, but the no-match gate answers zero, not best.
      return {0, DEFAULT_ZCUT_SOFTMAX_ALPHA_PRECISE, false};
    case M::Balanced:
    default:
      return {1, DEFAULT_ZCUT_SOFTMAX_ALPHA_BALANCED, true};
  }
}
// The per-mode band fraction f: per-index override -> constant.
float resolveFraction(const VectorIndex& vidx, int fractionMode) {
  constexpr float kConst[3] = {DEFAULT_ZCUT_FRACTION_PRECISE,
                               DEFAULT_ZCUT_FRACTION_BALANCED,
                               DEFAULT_ZCUT_FRACTION_BROAD};
  return vidx.zcutFractionDefault(fractionMode).value_or(kConst[fractionMode]);
}
}  // namespace

// ____________________________________________________________________________
CslsCut resolveCslsCut(const VectorSearchConfiguration& config,
                       const VectorIndex& vidx) {
  AD_CORRECTNESS_CHECK(config.hasCslsCut());
  CslsCut cut;
  if (config.cslsThreshold_.has_value()) {
    // The classic fixed cut: everything else stays at its (unused) default.
    cut.mode_ = CslsCut::Mode::Threshold;
    cut.threshold_ = config.cslsThreshold_.value();
    return cut;
  }
  using CutSignal = VectorSearchConfiguration::CutSignal;
  const CoverageParams cov = coverageParams(config.autoCut_.value());
  // The floor-estimator fraction, the exact no-match gate, and the top-anchored
  // widen depth (broad's fraction + a margin). Each: per-index `zcut*` serving
  // default -> constant. The widen depth uses the BROADEST fraction so the
  // cached reranked set holds everything any coverage mode could keep.
  cut.floorFraction_ =
      vidx.zcutFloorFractionDefault().value_or(DEFAULT_ZCUT_FLOOR_FRACTION);
  cut.gateZ_ = vidx.zcutGateZDefault().value_or(DEFAULT_ZCUT_GATE_Z);
  cut.widenFraction_ = resolveFraction(vidx, 2) + DEFAULT_ZCUT_WIDEN_MARGIN;
  // Bound the widening so a NO-MATCH query (whose window never separates from
  // the best) reranks at most a few rerank floors, not the whole index; hitting
  // it is logged, never silent.
  cut.rerankCap_ = std::max<size_t>(vidx.cslsRerankFloor() * 8, 10'000);
  cut.keepAtLeastOne_ = cov.keepAtLeastOne_;
  if (config.cutSignal_ == CutSignal::Softmax) {
    cut.mode_ = CslsCut::Mode::Softmax;
    // T precedence: per-query override -> runtime-config serving default ->
    // the BUILD-time corpus calibration (`.meta`) -> the constant fallback.
    const float calibratedOrConst =
        vidx.calibratedSoftmaxTemperature().value_or(
            DEFAULT_CSLS_SOFTMAX_TEMPERATURE);
    cut.temperature_ = config.softmaxTemperature_.value_or(
        vidx.softmaxTemperatureDefault().value_or(calibratedOrConst));
    const size_t neighbors =
        config.cslsNeighbors_.value_or(vidx.cslsNeighbors());
    cut.softmaxN_ = config.softmaxN_.value_or(vidx.softmaxNDefault().value_or(
        std::max<size_t>(DEFAULT_CSLS_SOFTMAX_N_FACTOR * neighbors, 1)));
    cut.alpha_ = cov.softmaxAlpha_;
  } else {
    // The top-anchored z-cut over the cosine or CSLS signal.
    cut.mode_ = CslsCut::Mode::ZCut;
    cut.signal_ = config.cutSignal_ == CutSignal::Csls
                      ? CslsCut::Signal::Csls
                      : CslsCut::Signal::Cosine;
    cut.fraction_ = resolveFraction(vidx, cov.fractionMode_);
  }
  return cut;
}

namespace {
// The process-lifetime cache of the MODE-INDEPENDENT reranked-to-plateau
// stage. Keyed by a string (index + query point + candidate identity + r(q)
// size + depth knobs); the value is the reranked candidate set that any
// coverage mode / signal then cuts in O(reranked).
struct CslsRerankedSizeGetter {
  ad_utility::MemorySize operator()(const CslsReranked& r) const {
    return ad_utility::MemorySize::bytes(
        r.entityBits_.size() * sizeof(uint64_t) +
        r.storeRows_.size() * sizeof(uint64_t) +
        r.fineDist_.size() * sizeof(float) + sizeof(CslsReranked));
  }
};
using CslsRerankedCache = ad_utility::ConcurrentCache<
    ad_utility::LRUCache<std::string, CslsReranked, CslsRerankedSizeGetter>>;
CslsRerankedCache& cslsRerankedCache() {
  static CslsRerankedCache cache{/*maxNumEntries=*/256,
                                 ad_utility::MemorySize::megabytes(512),
                                 ad_utility::MemorySize::megabytes(128)};
  return cache;
}

// The effective full-precision decision for a query: the per-query
// `vec:fullPrecision` flag OR the server-wide default env
// `QLEVER_VECTOR_SEARCH_FULL_PRECISION` (`vectorSearchFullPrecision()`). When
// true, the search bypasses the quantized coarse layer and scans the
// full-precision fine layer (e.g. bf16) exhaustively.
bool effectiveFullPrecision(const VectorSearchConfiguration& config) {
  return config.fullPrecision_ || vectorSearchFullPrecision();
}
}  // namespace

// ____________________________________________________________________________
std::vector<CslsScoredEntity> runCslsCut(
    const VectorIndex& vidx, const VectorSearchConfiguration& config,
    const CslsCut& cut, size_t neighbors, std::optional<Id> queryEntity,
    ql::span<const float> query, std::optional<ql::span<const Id>> candidates,
    std::string_view candidateIdentity,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored,
    bool* rerankCacheHit) {
  auto setScored = [&](size_t n) {
    if (numScored != nullptr) *numScored = n;
  };
  auto setHit = [&](bool h) {
    if (rerankCacheHit != nullptr) *rerankCacheHit = h;
  };
  const bool fullPrecision = effectiveFullPrecision(config);
  const qlever::vector::Bf16Kernel bf16Kernel = config.bf16Kernel_;
  // The fixed threshold (and any non-coverage cut) stays on the direct,
  // uncached path -- it is not what mode-switching re-applies.
  if (cut.mode_ != CslsCut::Mode::ZCut && cut.mode_ != CslsCut::Mode::Softmax) {
    setHit(false);
    return queryEntity.has_value()
               ? vidx.searchCslsByEntity(queryEntity.value(), cut, neighbors,
                                         candidates, config.maxDistance_,
                                         checkInterrupt, numScored,
                                         fullPrecision, bf16Kernel)
               : vidx.searchCsls(query, cut, neighbors, candidates,
                                 config.maxDistance_, checkInterrupt, numScored,
                                 fullPrecision, bf16Kernel);
  }
  // Cache key: the mode-independent rerank drivers. NOT the coverage mode,
  // signal, softmax params, cslsKCap, or maxDistance (all applied at cut time).
  // `fp=` IS part of the key: full precision changes WHICH candidates are
  // reranked (the whole fine layer vs. the coarse-preselected batch), so the
  // two modes must never share a cached reranked set.
  // `bf16k=` IS part of the key: the kernels agree only to ~1e-6, so a cached
  // reranked set from one kernel must not silently answer a query pinned to
  // another -- and it keeps an A/B benchmark's timings from being polluted by
  // cross-kernel cache hits.
  std::string key = absl::StrCat(
      "CSLS_RERANK idx=", config.indexName_, " nb=", neighbors,
      " ff=", absl::Hex(absl::bit_cast<uint32_t>(cut.floorFraction_)),
      " wf=", absl::Hex(absl::bit_cast<uint32_t>(cut.widenFraction_)),
      " cap=", cut.rerankCap_, " fp=", fullPrecision,
      " bf16k=", static_cast<int>(bf16Kernel));
  appendQueryPointToCacheKey(&key, config);
  absl::StrAppend(&key, " cand={", candidateIdentity, "}");
  auto compute = [&]() -> CslsReranked {
    return queryEntity.has_value()
               ? vidx.computeCslsRerankedByEntity(
                     queryEntity.value(), neighbors, cut.floorFraction_,
                     cut.widenFraction_, cut.rerankCap_, candidates,
                     checkInterrupt, fullPrecision, bf16Kernel)
               : vidx.computeCslsReranked(query, neighbors, cut.floorFraction_,
                                          cut.widenFraction_, cut.rerankCap_,
                                          candidates, checkInterrupt,
                                          fullPrecision, bf16Kernel);
  };
  auto res =
      cslsRerankedCache().computeOnce(key, compute, /*onlyReadFromCache=*/false,
                                      [](const CslsReranked&) { return true; });
  setHit(res._cacheStatus != ad_utility::CacheStatus::computed);
  const CslsReranked& reranked = *res._resultPointer;
  setScored(reranked.scored_);
  return vidx.applyCslsCut(reranked, cut, config.maxDistance_);
}

// ____________________________________________________________________________
size_t cslsRerankedCacheSizeForTesting() {
  return cslsRerankedCache().numNonPinnedEntries();
}
void clearCslsRerankedCacheForTesting() { cslsRerankedCache().clearAll(); }

// ____________________________________________________________________________
IdTable computeWholeIndexSearch(
    const VectorSearchConfiguration& config, QueryExecutionContext* qec,
    const ad_utility::SharedCancellationHandle& handle) {
  const Index& index = qec->getIndex();
  std::shared_ptr<const VectorIndex> vidx =
      getVectorIndex(index, config.indexName_);
  if (!vidx) {
    throw std::runtime_error{absl::StrCat(
        "There is no loaded vector index named '", config.indexName_,
        "'. Was the index built with `--service-index`?")};
  }
  if (config.hasCslsCut()) {
    validateCslsIsAvailable(config, *vidx);
  }

  const size_t width =
      1 + static_cast<size_t>(config.scoreVariable_.has_value()) +
      static_cast<size_t>(config.coarseScoreVariable_.has_value()) +
      static_cast<size_t>(config.cslsVariable_.has_value());
  IdTable idTable{width, qec->getAllocator()};

  // Resolve the query point (shared with the `vec:distance` function): an
  // explicit or embedded vector goes into `query`; a constant entity
  // (`vec:query <iri>`) is searched by its STORED vector directly (no
  // decode/re-encode through f32), tracked in `queryEntity`; an
  // unknown/vectorless entity -> empty result.
  QueryPoint queryPoint =
      resolveQueryPoint(config, *vidx, index.getImpl(), handle);
  if (std::holds_alternative<std::monostate>(queryPoint)) {
    return idTable;
  }
  std::vector<float> query;
  std::optional<Id> queryEntity;
  if (std::holds_alternative<Id>(queryPoint)) {
    queryEntity = std::get<Id>(queryPoint);
  } else {
    query = std::move(std::get<std::vector<float>>(queryPoint));
  }

  using Algo = VectorSearchConfiguration::Algorithm;
  if (config.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()) {
    throw std::runtime_error{
        absl::StrCat("Vector search requested the HNSW algorithm, but index '",
                     config.indexName_, "' has no HNSW structure.")};
  }
  auto checkInterrupt = [&handle]() { handle->throwIfCancelled(); };
  // `vec:fullPrecision` (or the server-wide env default): skip the coarse layer
  // and scan the full-precision fine layer exhaustively. Applies to both the
  // CSLS/autoCut path (via `runCslsCut`) and the plain top-k path below.
  const bool fullPrecision = effectiveFullPrecision(config);

  if (config.hasCslsCut()) {
    // CSLS-machinery cut over the WHOLE index -- the fixed tau
    // (`vec:cslsThreshold`) or a dynamic `vec:autoCut` (knee/softmax): every
    // live vector is scored (one FULL scan on the fine layer of a
    // single-layer index; a full COARSE scan plus a bounded, auto-widening
    // fine rerank on a two-layer one -- see `searchCslsBytes`), then the
    // resolved cut selects the survivors. `vec:bindScore` stays the (fine)
    // cosine distance; survivors sort by it.
    const CslsCut cut = resolveCslsCut(config, *vidx);
    const size_t neighbors =
        config.cslsNeighbors_.value_or(vidx->cslsNeighbors());
    ad_utility::Timer cslsTimer{ad_utility::Timer::Started};
    size_t numScored = 0;
    bool cacheHit = false;
    // Whole-index search: the candidate identity is empty.
    std::vector<CslsScoredEntity> hits = runCslsCut(
        *vidx, config, cut, neighbors, queryEntity, query, std::nullopt,
        /*candidateIdentity=*/"", checkInterrupt, &numScored, &cacheHit);
    const double ms = cslsTimer.value().count() / 1000.0;
    logVectorSearchPhase(
        config.indexName_,
        cacheHit ? "csls rerank (cache hit) + cut"
        : (vidx->hasRerankLayer() && !fullPrecision)
            ? "coarse scan + rerank (csls)"
            : (vidx->hasRerankLayer()
                   ? "full fine scan (csls; full-precision forced, coarse skipped)"
                   : "full fine scan (csls; coarse layer unused)"),
        ms, numScored);
    logVectorSearchPhase(config.indexName_,
                         absl::StrCat("csls filter: ", numScored,
                                      " candidates -> ", hits.size(), " kept"),
                         ms);
    // An EXPLICIT `vec:k` caps the survivors (they are already ascending by
    // cosine distance); without one ALL survivors are returned.
    if (config.cslsKCap_.has_value() &&
        hits.size() > config.cslsKCap_.value()) {
      hits.resize(config.cslsKCap_.value());
    }
    idTable.resize(hits.size());
    const bool withScoreCsls = config.scoreVariable_.has_value();
    // Column layout: `?result`[, `?score`][, `?csls`] (`vec:bindCoarseScore`
    // is rejected together with the csls cut at parse time).
    const size_t cslsCol = 1 + static_cast<size_t>(withScoreCsls);
    for (size_t i = 0; i < hits.size(); ++i) {
      idTable(i, 0) = hits[i].entity_;
      if (withScoreCsls) {
        idTable(i, 1) = Id::makeFromDouble(hits[i].distance_);
      }
      if (config.cslsVariable_.has_value()) {
        idTable(i, cslsCol) = Id::makeFromDouble(hits[i].csls_);
      }
    }
    return idTable;
  }

  // Whole-index search (by the stored vector for the entity form -- no
  // decode/re-encode through f32).
  std::vector<ScoredEntity> results;
  // Full-precision mode is an EXHAUSTIVE fine-layer scan, so it also disables
  // the HNSW probe (an exact brute force, no approximation).
  bool useHnsw =
      vidx->hasHnsw() && config.algorithm_ != Algo::Exact && !fullPrecision;
  // The coarse (scan-layer) distance per result entity, kept iff
  // `vec:bindCoarseScore` asked for it. On a single-layer index the two
  // layers coincide, so the fine distance doubles as the coarse one and this
  // map stays unused.
  ad_utility::HashMap<Id, float> coarseDistances;
  const bool withCoarseScore = config.coarseScoreVariable_.has_value();

  if (vidx->hasRerankLayer() && !fullPrecision) {
    // TWO-LAYER coarse-scan-then-rerank: (1) the coarse pass searches the
    // quantized scan matrix (brute force, or HNSW per `vec:algorithm` -- the
    // graph reads the scan bytes) for the top-`rerankK` candidates;
    // (2) the rerank pass recomputes their distances EXACTLY on the fine
    // rerank matrix and keeps the top `k`. `maxDistance` filters on the FINE
    // distance only -- the coarse pass must not drop near-boundary candidates
    // by their quantized distance. The default `rerankK` is scan-scalar
    // dependent (`defaultRerankK`): the 1-bit binary layer pre-ranks far more
    // coarsely than i8, so it keeps a wider candidate margin.
    const size_t rerankK =
        std::max(config.rerankK_.value_or(defaultRerankK(
                     vidx->metadata().config_.scalar_, config.k_)),
                 config.k_);
    ad_utility::Timer coarseTimer{ad_utility::Timer::Started};
    std::vector<ScoredEntity> coarse;
    if (queryEntity.has_value()) {
      coarse = useHnsw ? vidx->searchHnswByEntity(queryEntity.value(), rerankK,
                                                  std::nullopt, checkInterrupt)
                       : vidx->searchExactCoarseByEntity(
                             queryEntity.value(), rerankK, std::nullopt,
                             std::nullopt, checkInterrupt);
    } else {
      coarse = useHnsw ? vidx->searchHnsw(query, rerankK, std::nullopt,
                                          checkInterrupt)
                       : vidx->searchExactCoarse(query, rerankK, std::nullopt,
                                                 std::nullopt, checkInterrupt);
    }
    logVectorSearchPhase(
        config.indexName_,
        useHnsw ? "coarse HNSW probe" : "brute-force scan (coarse layer)",
        coarseTimer.value().count() / 1000.0,
        useHnsw ? std::optional<size_t>{}
                : std::optional<size_t>{vidx->numLiveVectors()});
    std::vector<Id> candidates;
    candidates.reserve(coarse.size());
    for (const auto& hit : coarse) {
      candidates.push_back(hit.entity_);
      if (withCoarseScore) {
        coarseDistances[hit.entity_] = hit.distance_;
      }
    }
    // Fine pass: exact distances over exactly the coarse candidates (the
    // restricted `searchExact` merge-joins them against the id-sorted store).
    ad_utility::Timer rerankTimer{ad_utility::Timer::Started};
    results = queryEntity.has_value()
                  ? vidx->searchExactByEntity(queryEntity.value(), config.k_,
                                              candidates, config.maxDistance_,
                                              checkInterrupt,
                                              /*numScored=*/nullptr,
                                              config.bf16Kernel_)
                  : vidx->searchExact(query, config.k_, candidates,
                                      config.maxDistance_, checkInterrupt,
                                      /*numScored=*/nullptr, config.bf16Kernel_);
    logVectorSearchPhase(config.indexName_, "rerank (fine layer)",
                         rerankTimer.value().count() / 1000.0,
                         candidates.size());
  } else {
    ad_utility::Timer scanTimer{ad_utility::Timer::Started};
    if (queryEntity.has_value()) {
      results =
          useHnsw
              ? vidx->searchHnswByEntity(queryEntity.value(), config.k_,
                                         config.maxDistance_, checkInterrupt)
              : vidx->searchExactByEntity(queryEntity.value(), config.k_,
                                          std::nullopt, config.maxDistance_,
                                          checkInterrupt, /*numScored=*/nullptr,
                                          config.bf16Kernel_);
    } else {
      results = useHnsw
                    ? vidx->searchHnsw(query, config.k_, config.maxDistance_,
                                       checkInterrupt)
                    : vidx->searchExact(query, config.k_, std::nullopt,
                                        config.maxDistance_, checkInterrupt,
                                        /*numScored=*/nullptr,
                                        config.bf16Kernel_);
    }
    logVectorSearchPhase(
        config.indexName_,
        useHnsw ? "HNSW probe"
                : (vidx->hasRerankLayer()
                       ? "brute-force scan (full-precision, coarse layer skipped)"
                       : "brute-force scan"),
        scanTimer.value().count() / 1000.0,
        useHnsw ? std::optional<size_t>{}
                : std::optional<size_t>{vidx->numLiveVectors()});
  }

  idTable.resize(results.size());
  bool withScore = config.scoreVariable_.has_value();
  const size_t coarseCol = withScore ? 2 : 1;
  for (size_t i = 0; i < results.size(); ++i) {
    idTable(i, 0) = results[i].entity_;
    if (withScore) {
      idTable(i, 1) = Id::makeFromDouble(results[i].distance_);
    }
    if (withCoarseScore) {
      // Every fine result came out of the coarse candidate set, so the lookup
      // always hits on a two-layer index; on a single-layer index coarse ==
      // fine by definition.
      auto it = coarseDistances.find(results[i].entity_);
      idTable(i, coarseCol) = Id::makeFromDouble(
          it != coarseDistances.end() ? it->second : results[i].distance_);
    }
  }
  return idTable;
}

}  // namespace qlever::vector

// ____________________________________________________________________________
VectorSearch::VectorSearch(QueryExecutionContext* qec,
                           qlever::vector::VectorSearchConfiguration config)
    : Operation{qec}, config_{std::move(config)} {
  // Column layout: `?result`, then the optional fine `?score`, then the
  // optional coarse `?coarseScore`, then the optional `?csls` (each occupying
  // the next column iff present).
  size_t next = 0;
  variableColumns_[config_.resultVariable_] =
      makeAlwaysDefinedColumn(ColumnIndex{next++});
  if (config_.scoreVariable_.has_value()) {
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{next++});
  }
  if (config_.coarseScoreVariable_.has_value()) {
    variableColumns_[config_.coarseScoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{next++});
  }
  if (config_.cslsVariable_.has_value()) {
    variableColumns_[config_.cslsVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{next++});
  }
}

// ____________________________________________________________________________
std::string VectorSearch::getDescriptor() const {
  return absl::StrCat("VectorSearch on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearch::getResultWidth() const {
  return 1 + static_cast<size_t>(config_.scoreVariable_.has_value()) +
         static_cast<size_t>(config_.coarseScoreVariable_.has_value()) +
         static_cast<size_t>(config_.cslsVariable_.has_value());
}

// ____________________________________________________________________________
float VectorSearch::getMultiplicity(size_t) { return 1.0f; }

// ____________________________________________________________________________
uint64_t VectorSearch::getSizeEstimateBeforeLimit() { return config_.k_; }

// ____________________________________________________________________________
size_t VectorSearch::getCostEstimate() {
  // Whole-index search: a brute-force scan touches every vector, an HNSW probe
  // roughly log(N) of them. The index may not be loaded while planning (e.g.
  // in tests); fall back to a small constant then.
  auto vidx = qlever::vector::getVectorIndex(getExecutionContext()->getIndex(),
                                             config_.indexName_);
  if (!vidx) {
    return config_.k_;
  }
  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  size_t n = vidx->numVectors();
  if (useHnsw) {
    return static_cast<size_t>(std::log2(static_cast<double>(n) + 1) + 1) *
           config_.k_;
  }
  return n;
}

// ____________________________________________________________________________
VariableToColumnMap VectorSearch::computeVariableToColumnMap() const {
  return variableColumns_;
}

// ____________________________________________________________________________
std::string VectorSearch::getCacheKeyImpl() const {
  std::string key = absl::StrCat(
      "VECTOR_SEARCH index=", config_.indexName_, " k=", config_.k_,
      " algo=", static_cast<int>(config_.algorithm_),
      " score=", config_.scoreVariable_.has_value(),
      " coarseScore=", config_.coarseScoreVariable_.has_value());
  // Execution knobs that change the result (fullPrecision skips the coarse
  // layer -> a different, exact top-k) or that must keep an A/B benchmark from
  // reading another variant's cached result (bf16Kernel). The CSLS score cache
  // already keys on both; the main op cache MUST too.
  absl::StrAppend(&key, " fullPrecision=", config_.fullPrecision_,
                  " bf16Kernel=", static_cast<int>(config_.bf16Kernel_));
  if (config_.rerankK_.has_value()) {
    absl::StrAppend(&key, " rerankK=", config_.rerankK_.value());
  }
  if (config_.maxDistance_.has_value()) {
    // Bit-exact so that two nearby values never share a key.
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  qlever::vector::appendCslsCutToCacheKey(&key, config_);
  qlever::vector::appendQueryPointToCacheKey(&key, config_);
  return key;
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorSearch::cloneImpl() const {
  return std::make_unique<VectorSearch>(getExecutionContext(), config_);
}

// ____________________________________________________________________________
Result VectorSearch::computeResult([[maybe_unused]] bool requestLaziness) {
  return {qlever::vector::computeWholeIndexSearch(
              config_, getExecutionContext(), cancellationHandle_),
          resultSortedOn(), LocalVocab{}};
}
