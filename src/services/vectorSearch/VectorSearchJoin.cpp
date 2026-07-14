// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSearchJoin.h"

#include <absl/base/casts.h>
#include <absl/strings/str_cat.h>

#include <cmath>

#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorQueryPoint.h"
#include "services/vectorSearch/VectorSearch.h"
#include "util/HashMap.h"
#include "util/HashSet.h"
#include "util/Timer.h"

// ____________________________________________________________________________
VectorSearchJoin::VectorSearchJoin(
    QueryExecutionContext* qec,
    qlever::vector::VectorSearchConfiguration config,
    std::shared_ptr<QueryExecutionTree> child)
    : Operation{qec}, config_{std::move(config)}, child_{std::move(child)} {
  AD_CONTRACT_CHECK(config_.leftVariable_.has_value(),
                    "VectorSearchJoin requires a `<candidates>` variable.");
  // The parser guarantees these; direct constructions must obey them too.
  AD_CONTRACT_CHECK(config_.hasQueryPoint() || !annotatesCandidatesInPlace(),
                    "`<candidates> ?in` with `<result> ?in` (annotate in "
                    "place) requires a query point.");
  AD_CONTRACT_CHECK(
      config_.hasQueryPoint() || !config_.coarseScoreVariable_.has_value(),
      "`<bindCoarseScore>` requires a query point.");
  AD_CONTRACT_CHECK(config_.hasQueryPoint() || !config_.hasCslsCut(),
                    "`<cslsThreshold>`/`<autoCut>` requires a query point.");
  if (!child_) {
    // INCOMPLETE form: the subtree binding `<candidates>` comes from the
    // surrounding query. Expose the variables as possibly-undefined columns
    // so that the join enumeration can find the connection (see
    // `IncompleteJoinOperation`); `addJoinChild` builds the completed
    // operation. If nothing binds `<candidates>`, the leaf itself executes
    // FORM W (see `computeResult`).
    size_t next = 0;
    variableColumns_[config_.leftVariable_.value()] =
        makePossiblyUndefinedColumn(ColumnIndex{next++});
    if (!annotatesCandidatesInPlace()) {
      variableColumns_[config_.resultVariable_] =
          makePossiblyUndefinedColumn(ColumnIndex{next++});
    }
    if (config_.scoreVariable_.has_value()) {
      variableColumns_[config_.scoreVariable_.value()] =
          makePossiblyUndefinedColumn(ColumnIndex{next++});
    }
    if (config_.coarseScoreVariable_.has_value()) {
      variableColumns_[config_.coarseScoreVariable_.value()] =
          makePossiblyUndefinedColumn(ColumnIndex{next++});
    }
    if (config_.cslsVariable_.has_value()) {
      variableColumns_[config_.cslsVariable_.value()] =
          makePossiblyUndefinedColumn(ColumnIndex{next++});
    }
    return;
  }
  const auto& childCols = child_->getVariableColumns();
  auto it = childCols.find(config_.leftVariable_.value());
  if (it == childCols.end()) {
    throw std::runtime_error{
        absl::StrCat("The vector-search `<candidates>`/`<left>` variable ",
                     config_.leftVariable_.value().name(),
                     " is not bound by the join child.")};
  }
  leftCol_ = it->second.columnIndex_;

  // The `?result`/`?score`/`?coarseScore` columns are appended to the child's
  // columns (except in the annotate form, where `?result` IS the candidates
  // column); if the child also binds one of them, silently overwriting the
  // mapping would relabel a child column and return wrong results -- reject.
  auto checkNotBoundByChild = [&childCols](const Variable& var,
                                           std::string_view parameter) {
    if (childCols.contains(var)) {
      throw std::runtime_error{absl::StrCat(
          "The vector-search `<", parameter, ">` variable ", var.name(),
          " must not be bound by the subtree that binds `<candidates>`.")};
    }
  };
  if (!annotatesCandidatesInPlace()) {
    checkNotBoundByChild(config_.resultVariable_, "result");
  }
  if (config_.scoreVariable_.has_value()) {
    checkNotBoundByChild(config_.scoreVariable_.value(), "bindScore");
  }
  if (config_.coarseScoreVariable_.has_value()) {
    checkNotBoundByChild(config_.coarseScoreVariable_.value(),
                         "bindCoarseScore");
  }
  if (config_.cslsVariable_.has_value()) {
    checkNotBoundByChild(config_.cslsVariable_.value(), "bindCsls");
  }

  // Output columns: all child columns, then `?result` (unless it annotates
  // the candidates column in place), then the optional `?score`,
  // `?coarseScore`, and `?csls`.
  variableColumns_ = childCols;
  size_t next = child_->getResultWidth();
  if (!annotatesCandidatesInPlace()) {
    variableColumns_[config_.resultVariable_] =
        makeAlwaysDefinedColumn(ColumnIndex{next++});
  }
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
std::string VectorSearchJoin::multipleJoinVariablesError() const {
  return absl::StrCat(
      "A vector search whose join variable ",
      config_.leftVariable_.value().name(),
      " comes from the surrounding query must be connected to the rest of the "
      "query through that variable only; it also shares one of its result "
      "variables. Rename the shared result variable, or bind the extra "
      "connection in a separate pattern.");
}

// ____________________________________________________________________________
std::shared_ptr<Operation> VectorSearchJoin::addJoinChild(
    std::shared_ptr<QueryExecutionTree> child, const Variable& var) const {
  AD_CONTRACT_CHECK(var == config_.leftVariable_.value());
  auto completed = std::make_shared<VectorSearchJoin>(
      getExecutionContext(), config_, std::move(child));
  // Carry over any warnings accrued on the incomplete operation (mirrors
  // `SpatialJoin::addChild`).
  for (const auto& warning : *getWarnings().rlock()) {
    completed->addWarning(warning);
  }
  return completed;
}

// ____________________________________________________________________________
std::string VectorSearchJoin::getDescriptor() const {
  return absl::StrCat("VectorSearchJoin on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearchJoin::getResultWidth() const {
  size_t ownColumns =
      static_cast<size_t>(!annotatesCandidatesInPlace()) +
      static_cast<size_t>(config_.scoreVariable_.has_value()) +
      static_cast<size_t>(config_.coarseScoreVariable_.has_value()) +
      static_cast<size_t>(config_.cslsVariable_.has_value());
  if (!child_) {
    // Incomplete: `<candidates>`, plus the own columns.
    return 1 + ownColumns;
  }
  return child_->getResultWidth() + ownColumns;
}

// ____________________________________________________________________________
float VectorSearchJoin::getMultiplicity(size_t col) {
  if (!child_) {
    return 1.0f;
  }
  // Passthrough columns inherit the child's multiplicity -- scaled by the
  // k-fold expansion of FORM E; FORM P keeps at most the child's rows. The
  // synthetic result/score columns are treated as ~unique.
  size_t childWidth = child_->getResultWidth();
  if (col < childWidth) {
    return config_.hasQueryPoint()
               ? child_->getMultiplicity(col)
               : child_->getMultiplicity(col) * static_cast<float>(config_.k_);
  }
  return 1.0f;
}

// ____________________________________________________________________________
uint64_t VectorSearchJoin::getSizeEstimateBeforeLimit() {
  if (!child_) {
    // FORM W (never completed): a whole-index top-k.
    return config_.k_;
  }
  if (config_.hasQueryPoint()) {
    // FORM P never leaves the bound set: at most the child's rows, and at
    // most ~k of its distinct candidates unless all are kept.
    if (!config_.keepAllCandidates_) {
      return std::min<uint64_t>(child_->getSizeEstimate(), config_.k_);
    }
    // Annotate form: exactly one row per child row whose candidate is a
    // MEMBER -- the rowmap merge drops candidates with no stored vector. So
    // the surviving DISTINCT candidates are capped at the index's live-vector
    // count; scaled by the child's multiplicity on the candidate column, that
    // bounds the output rows. This is tighter than the raw child size whenever
    // the candidate set holds many non-members, which is the whole point of a
    // pre-filter over a metadata subset.
    uint64_t childEstimate = child_->getSizeEstimate();
    auto vidx = qlever::vector::getVectorIndex(
        getExecutionContext()->getIndex(), config_.indexName_);
    if (vidx == nullptr) {
      return childEstimate;
    }
    double candidateMultiplicity =
        std::max(1.0f, child_->getMultiplicity(leftCol_));
    auto memberBound = static_cast<uint64_t>(
        static_cast<double>(vidx->numLiveVectors()) * candidateMultiplicity);
    return std::min<uint64_t>(childEstimate, memberBound);
  }
  // FORM E: k result rows per child row.
  return child_->getSizeEstimate() * config_.k_;
}

// ____________________________________________________________________________
size_t VectorSearchJoin::getCostEstimate() {
  // FORM E: the child's own cost plus one index probe per child row. A probe
  // is ~log(N) with HNSW, but a full scan of the N vectors without it.
  auto vidx = qlever::vector::getVectorIndex(getExecutionContext()->getIndex(),
                                             config_.indexName_);
  size_t probeCost = config_.k_;
  if (vidx) {
    using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
    bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
    size_t n = vidx->numVectors();
    probeCost =
        useHnsw
            ? static_cast<size_t>(std::log2(static_cast<double>(n) + 1) + 1) *
                  config_.k_
            : n;
  }
  if (!child_) {
    return probeCost;
  }
  if (config_.hasQueryPoint()) {
    // FORM P: ONE exact scan restricted to the child's candidate set (plus
    // the gather over the child's rows), not a probe per row.
    return child_->getCostEstimate() + child_->getSizeEstimate();
  }
  return child_->getCostEstimate() + child_->getSizeEstimate() * probeCost;
}

// ____________________________________________________________________________
VariableToColumnMap VectorSearchJoin::computeVariableToColumnMap() const {
  return variableColumns_;
}

// ____________________________________________________________________________
std::string VectorSearchJoin::getCacheKeyImpl() const {
  std::string key = absl::StrCat(
      "VECTOR_SEARCH_JOIN index=", config_.indexName_, " k=", config_.k_,
      " keepAll=", config_.keepAllCandidates_,
      " algo=", static_cast<int>(config_.algorithm_), " leftCol=", leftCol_,
      " annotate=", annotatesCandidatesInPlace(),
      " score=", config_.scoreVariable_.has_value(),
      " coarseScore=", config_.coarseScoreVariable_.has_value());
  // See VectorSearch::getCacheKeyImpl: fullPrecision changes the result and
  // bf16Kernel must isolate an A/B run from cross-variant cache hits.
  absl::StrAppend(&key, " fullPrecision=", config_.fullPrecision_,
                  " bf16Kernel=", static_cast<int>(config_.bf16Kernel_),
                  " i8Kernel=", static_cast<int>(config_.i8Kernel_));
  if (config_.rerankK_.has_value()) {
    absl::StrAppend(&key, " rerankK=", config_.rerankK_.value());
  }
  if (config_.maxDistance_.has_value()) {
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  qlever::vector::appendCslsCutToCacheKey(&key, config_);
  qlever::vector::appendQueryPointToCacheKey(&key, config_);
  if (child_) {
    absl::StrAppend(&key, " {", child_->getCacheKey(), "}");
  } else {
    // A leaf that the planner never completed either executes FORM W (query
    // point present) or throws; the marker keeps its key distinct from a
    // completed operation's.
    absl::StrAppend(&key, " INCOMPLETE");
  }
  return key;
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorSearchJoin::cloneImpl() const {
  return std::make_unique<VectorSearchJoin>(getExecutionContext(), config_,
                                            child_ ? child_->clone() : nullptr);
}

// ____________________________________________________________________________
Result VectorSearchJoin::computeResult([[maybe_unused]] bool requestLaziness) {
  if (!child_) {
    // The planner found NO subtree binding `<candidates>` -- the variable is
    // unbound by the surrounding query.
    if (!config_.hasQueryPoint()) {
      // FORM E without candidates is genuinely malformed: there are no query
      // entities and nothing else to search for.
      throw std::runtime_error{absl::StrCat(
          "The vector-search `<candidates>`/`<left>` variable ",
          config_.leftVariable_.value().name(),
          " is not bound anywhere: bind it in the surrounding query (each "
          "candidate is then searched by its own stored vector), or add an "
          "explicit query point (`<queryVector>`, `<query>`, `<queryText>`, "
          "or `<imageUrl>`/`<imageBase64>`) to search the whole index.")};
    }
    if (!annotatesCandidatesInPlace()) {
      throw std::runtime_error{absl::StrCat(
          "The vector-search `<candidates>` variable ",
          config_.leftVariable_.value().name(),
          " is not bound anywhere: with a query point and unbound "
          "`<candidates>` the search runs over the WHOLE index and binds its "
          "matches to that variable itself, so `<result>` must name the SAME "
          "variable (or omit `<candidates>` altogether), but it is ",
          config_.resultVariable_.name(), ".")};
    }
    // FORM W: identical to omitting `<candidates>` -- the whole-index top-k,
    // bound to the (shared) `<candidates>`/`<result>` variable.
    return {qlever::vector::computeWholeIndexSearch(
                config_, getExecutionContext(), cancellationHandle_),
            resultSortedOn(), LocalVocab{}};
  }
  const Index& index = getExecutionContext()->getIndex();
  std::shared_ptr<const qlever::vector::VectorIndex> vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  if (!vidx) {
    throw std::runtime_error{absl::StrCat(
        "There is no loaded vector index named '", config_.indexName_,
        "'. Was the index built with `--service-index`?")};
  }

  std::shared_ptr<const Result> childRes = child_->getResult();
  const IdTable& childTable = childRes->idTable();
  IdTable result{getResultWidth(), getExecutionContext()->getAllocator()};

  if (config_.hasQueryPoint()) {
    // FORM P (PRE-FILTER): score the bound candidates against the fixed
    // query point; the search never leaves the bound set.
    computePreFilterRows(*vidx, childTable, &result);
    return {std::move(result), resultSortedOn(),
            childRes->getCopyOfLocalVocab()};
  }

  // FORM E (ENTITY-TO-ENTITY): for each bound candidate, the k nearest of
  // its OWN stored vector.
  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  if (config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()) {
    throw std::runtime_error{
        absl::StrCat("Vector search requested the HNSW algorithm, but index '",
                     config_.indexName_, "' has no HNSW structure.")};
  }
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  bool withScore = config_.scoreVariable_.has_value();
  auto checkInterrupt = [this]() { checkCancellation(); };
  const size_t childWidth = childTable.numColumns();

  // Children are often sorted (or at least clustered) by the join column, so
  // memoize the search results per distinct query entity -- without this, a
  // child with many duplicate `<candidates>` values re-runs identical
  // searches.
  ad_utility::HashMap<uint64_t, std::vector<qlever::vector::ScoredEntity>>
      hitsByEntity;

  for (size_t row = 0; row < childTable.numRows(); ++row) {
    checkCancellation();
    Id leftId = childTable(row, leftCol_);
    auto [memo, isNew] = hitsByEntity.try_emplace(leftId.getBits());
    if (isNew && vidx->hasVector(leftId)) {
      // Search by the STORED vector (no decode/re-encode through f32).
      memo->second =
          useHnsw
              ? vidx->searchHnswByEntity(leftId, config_.k_,
                                         config_.maxDistance_, checkInterrupt)
              : vidx->searchExactByEntity(leftId, config_.k_, std::nullopt,
                                          config_.maxDistance_, checkInterrupt);
    }
    // A query entity without a vector contributes no result rows (UNDEF and
    // unknown entities behave the same way).
    for (const auto& hit : memo->second) {
      result.emplace_back();
      size_t outRow = result.numRows() - 1;
      for (size_t c = 0; c < childWidth; ++c) {
        result(outRow, c) = childTable(row, c);
      }
      result(outRow, childWidth) = hit.entity_;
      if (withScore) {
        result(outRow, childWidth + 1) = Id::makeFromDouble(hit.distance_);
      }
    }
  }
  return {std::move(result), resultSortedOn(), childRes->getCopyOfLocalVocab()};
}

// ____________________________________________________________________________
void VectorSearchJoin::computePreFilterRows(
    const qlever::vector::VectorIndex& vidx, const IdTable& childTable,
    IdTable* result) {
  // FORM P: every bound candidate is scored by the exact distance of its
  // STORED vector to the FIXED query point (brute force over exactly the
  // candidate set; on a two-layer index a coarse scan over the set prunes to
  // the top-`rerankK` and the fine layer reranks). `?in == ?out` annotates
  // the child's rows in place; a distinct `?out` appends the top-k of the
  // bound set as a fresh column. Rows whose candidate is not kept -- no
  // stored vector, cut by `k`, or beyond `maxDistance` -- are dropped.
  qlever::vector::QueryPoint queryPoint = qlever::vector::resolveQueryPoint(
      config_, vidx, getExecutionContext()->getIndex().getImpl(),
      cancellationHandle_);
  if (std::holds_alternative<std::monostate>(queryPoint)) {
    // Unknown or vectorless constant query entity -> no distances -> empty.
    return;
  }
  std::optional<Id> queryEntity;
  std::vector<float> query;
  if (std::holds_alternative<Id>(queryPoint)) {
    queryEntity = std::get<Id>(queryPoint);
  } else {
    query = std::move(std::get<std::vector<float>>(queryPoint));
  }
  auto checkInterrupt = [this]() { checkCancellation(); };

  // All bound candidate ids, WITH duplicates (a photo bound to several works
  // repeats). No hash-set dedup here: the restricted `searchExact*` below sorts
  // the ids and the rowmap merge-join emits each matched row exactly ONCE (on a
  // match it advances past that id, so a repeated candidate is skipped) -- so
  // the merge both DEDUPS and drops candidates that have no stored vector, for
  // free. When the candidates already arrive sorted (e.g. via `vec:hasMember`
  // or an index scan) even the sort is skipped.
  std::vector<Id> candidates;
  candidates.reserve(childTable.numRows());
  for (size_t row = 0; row < childTable.numRows(); ++row) {
    candidates.push_back(childTable(row, leftCol_));
  }
  const size_t effectiveK =
      config_.keepAllCandidates_ ? candidates.size() : config_.k_;
  // The annotate form must score and return EVERY bound candidate, so it may
  // not be truncated by the engine's hard `MAX_SEARCH_RESULTS` top-k cap
  // (`keepAll` lifts the cap on every primitive of this path; the result stays
  // bounded by the already-materialized candidate table). A genuine top-k
  // (explicit `vec:k`, or a distinct `<result>` variable) keeps the cap.
  const bool keepAll = config_.keepAllCandidates_;

  const bool withCoarseScore = config_.coarseScoreVariable_.has_value();
  ad_utility::HashMap<Id, float> coarseDistances;
  // The CSLS value of every kept candidate, iff a csls cut binding it is set.
  ad_utility::HashMap<Id, float> cslsValues;
  std::vector<qlever::vector::ScoredEntity> scored;
  // `vec:fullPrecision` (or the server-wide env default): skip the quantized
  // coarse layer and score every bound candidate directly on the full-precision
  // fine layer (e.g. bf16) -- i.e. fall through to the single-layer `searchExact`
  // branch below, even on a two-layer index. (CSLS handles it inside
  // `runCslsCut`.)
  const bool fullPrecision =
      config_.fullPrecision_ || qlever::vector::vectorSearchFullPrecision();
  if (config_.hasCslsCut()) {
    // CSLS-machinery cut over the BOUND set -- the fixed tau
    // (`vec:cslsThreshold`) or a dynamic `vec:autoCut` (knee/softmax): every
    // candidate member is scored (a full fine-layer sweep on a single-layer
    // index; a full COARSE sweep plus a bounded, auto-widening fine rerank on
    // a two-layer one -- see `searchCslsBytes`), r(q) from the
    // top-`cslsNeighbors` of that set, then the resolved cut selects the
    // survivors. They keep their (fine) COSINE distance as the score; `vec:k`
    // (if explicitly given) caps them, otherwise all survivors are kept
    // (variable cardinality).
    qlever::vector::validateCslsIsAvailable(config_, vidx);
    const qlever::vector::CslsCut cut =
        qlever::vector::resolveCslsCut(config_, vidx);
    const size_t neighbors =
        config_.cslsNeighbors_.value_or(vidx.cslsNeighbors());
    ad_utility::Timer cslsTimer{ad_utility::Timer::Started};
    size_t numScored = 0;
    bool cacheHit = false;
    // The candidate set's cache identity is the child subtree's cache key (NOT
    // the millions of materialized ids), so a mode switch over the SAME bound
    // set hits the cached reranked stage. `leftCol_` MUST be part of it: the
    // same child can bind DIFFERENT candidate columns (e.g. `?photo` vs
    // `?painting`), which are different candidate sets under one child key --
    // without it the second query is served the first column's reranked set.
    const std::string candidateIdentity =
        child_ ? absl::StrCat(child_->getCacheKey(), " leftCol=", leftCol_)
               : std::string{"INCOMPLETE"};
    auto hits = qlever::vector::runCslsCut(
        vidx, config_, cut, neighbors, queryEntity, query,
        std::optional<ql::span<const Id>>{candidates}, candidateIdentity,
        checkInterrupt, &numScored, &cacheHit);
    const double ms = cslsTimer.value().count() / 1000.0;
    qlever::vector::logVectorSearchPhase(
        config_.indexName_,
        cacheHit ? "csls rerank (cache hit) + cut (index members)"
        : (vidx.hasRerankLayer() && !fullPrecision)
            ? "coarse scan + rerank (csls, index members)"
            : (vidx.hasRerankLayer()
                   ? "full fine scan (csls, index members; full-precision "
                     "forced, coarse skipped)"
                   : "full fine scan (csls, index members)"),
        ms, numScored);
    qlever::vector::logVectorSearchPhase(
        config_.indexName_,
        absl::StrCat("csls filter: ", numScored, " candidates -> ", hits.size(),
                     " kept"),
        ms);
    if (config_.cslsKCap_.has_value() &&
        hits.size() > config_.cslsKCap_.value()) {
      hits.resize(config_.cslsKCap_.value());
    }
    scored.reserve(hits.size());
    for (const auto& hit : hits) {
      scored.push_back(
          qlever::vector::ScoredEntity{hit.entity_, hit.distance_});
      cslsValues[hit.entity_] = hit.csls_;
    }
  } else if (vidx.hasRerankLayer() && !fullPrecision) {
    // Coarse pass over exactly the bound set, then the fine rerank pass over
    // the coarse survivors (never below `effectiveK`, so the annotate form
    // without `<k>` keeps every candidate). `maxDistance` filters on the
    // FINE distance only. The default `rerankK` is scan-scalar dependent
    // (`defaultRerankK`): the 1-bit binary layer pre-ranks far more coarsely
    // than i8, so it keeps a wider candidate margin.
    const size_t rerankK =
        std::max(config_.rerankK_.value_or(qlever::vector::defaultRerankK(
                     vidx.metadata().config_.scalar_, effectiveK)),
                 effectiveK);
    ad_utility::Timer coarseTimer{ad_utility::Timer::Started};
    size_t coarseScored = 0;
    // The coarse pass returns each survivor's STORE ROW (`ScoredRow`) so the
    // fine rerank below can score exactly those rows without a second
    // O(numVectors) `.rowmap` merge-join to recover them from the entity ids.
    auto coarse =
        queryEntity.has_value()
            ? vidx.searchExactCoarseByEntityWithRows(
                  queryEntity.value(), rerankK, candidates, std::nullopt,
                  checkInterrupt, &coarseScored, keepAll, config_.i8Kernel_)
            : vidx.searchExactCoarseWithRows(
                  query, rerankK, candidates, std::nullopt, checkInterrupt,
                  &coarseScored, keepAll, config_.i8Kernel_);
    // `coarseScored` = candidates that HAVE a vector (were actually scored),
    // not the raw candidate count -- the merge-join skips the vectorless ones.
    qlever::vector::logVectorSearchPhase(
        config_.indexName_, "brute-force scan (coarse layer, index members)",
        coarseTimer.value().count() / 1000.0, coarseScored);
    if (withCoarseScore) {
      for (const auto& hit : coarse) {
        coarseDistances[hit.entity_] = hit.distance_;
      }
    }
    ad_utility::Timer rerankTimer{ad_utility::Timer::Started};
    ql::span<const qlever::vector::ScoredRow> prunedRows{coarse};
    scored =
        queryEntity.has_value()
            ? vidx.searchExactByRowsByEntity(queryEntity.value(), effectiveK,
                                             prunedRows, config_.maxDistance_,
                                             checkInterrupt, config_.bf16Kernel_,
                                             keepAll)
            : vidx.searchExactByRows(query, effectiveK, prunedRows,
                                     config_.maxDistance_, checkInterrupt,
                                     config_.bf16Kernel_, keepAll);
    qlever::vector::logVectorSearchPhase(
        config_.indexName_, "rerank (fine layer)",
        rerankTimer.value().count() / 1000.0, coarse.size());
  } else {
    ad_utility::Timer scanTimer{ad_utility::Timer::Started};
    size_t numScored = 0;
    scored =
        queryEntity.has_value()
            ? vidx.searchExactByEntity(
                  queryEntity.value(), effectiveK, candidates,
                  config_.maxDistance_, checkInterrupt, &numScored,
                  config_.bf16Kernel_, keepAll, config_.i8Kernel_)
            : vidx.searchExact(query, effectiveK, candidates,
                               config_.maxDistance_, checkInterrupt, &numScored,
                               config_.bf16Kernel_, keepAll, config_.i8Kernel_);
    // Members actually scored, not the raw candidate count (see above).
    // When the index HAS a rerank layer but we reached this branch, it is
    // because `vec:fullPrecision` forced the coarse layer to be skipped.
    qlever::vector::logVectorSearchPhase(
        config_.indexName_,
        vidx.hasRerankLayer()
            ? "brute-force scan (fine layer, full-precision, index members)"
            : "brute-force scan (index members)",
        scanTimer.value().count() / 1000.0, numScored);
  }

  // The fine distance of every KEPT candidate.
  ad_utility::HashMap<Id, float> fineDistances;
  for (const auto& hit : scored) {
    fineDistances[hit.entity_] = hit.distance_;
  }

  // Emit the child's rows whose candidate was kept, appending the fresh
  // `?result` column (unless annotating in place) and the score column(s).
  const bool annotate = annotatesCandidatesInPlace();
  const bool withScore = config_.scoreVariable_.has_value();
  const bool withCsls = config_.cslsVariable_.has_value();
  const size_t childWidth = childTable.numColumns();
  for (size_t row = 0; row < childTable.numRows(); ++row) {
    checkCancellation();
    Id candidate = childTable(row, leftCol_);
    auto it = fineDistances.find(candidate);
    if (it == fineDistances.end()) {
      continue;
    }
    result->emplace_back();
    size_t outRow = result->numRows() - 1;
    for (size_t c = 0; c < childWidth; ++c) {
      (*result)(outRow, c) = childTable(row, c);
    }
    size_t next = childWidth;
    if (!annotate) {
      (*result)(outRow, next++) = candidate;
    }
    if (withScore) {
      (*result)(outRow, next++) = Id::makeFromDouble(it->second);
    }
    if (withCoarseScore) {
      // On a single-layer index the two layers coincide.
      auto coarseIt = coarseDistances.find(candidate);
      (*result)(outRow, next++) = Id::makeFromDouble(
          coarseIt != coarseDistances.end() ? coarseIt->second : it->second);
    }
    if (withCsls) {
      // Every kept candidate came out of the csls cut, so the lookup always
      // hits (`vec:bindCsls` requires `vec:cslsThreshold` at parse time).
      auto cslsIt = cslsValues.find(candidate);
      AD_CORRECTNESS_CHECK(cslsIt != cslsValues.end());
      (*result)(outRow, next++) = Id::makeFromDouble(cslsIt->second);
    }
  }
}
