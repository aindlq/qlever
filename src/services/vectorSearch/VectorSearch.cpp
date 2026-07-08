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
#include "util/HashMap.h"

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

  const size_t width =
      1 + static_cast<size_t>(config.scoreVariable_.has_value()) +
      static_cast<size_t>(config.coarseScoreVariable_.has_value());
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

  // Whole-index search (by the stored vector for the entity form -- no
  // decode/re-encode through f32).
  std::vector<ScoredEntity> results;
  bool useHnsw = vidx->hasHnsw() && config.algorithm_ != Algo::Exact;
  // The coarse (scan-layer) distance per result entity, kept iff
  // `vec:bindCoarseScore` asked for it. On a single-layer index the two
  // layers coincide, so the fine distance doubles as the coarse one and this
  // map stays unused.
  ad_utility::HashMap<Id, float> coarseDistances;
  const bool withCoarseScore = config.coarseScoreVariable_.has_value();

  if (vidx->hasRerankLayer()) {
    // TWO-LAYER coarse-scan-then-rerank: (1) the coarse pass searches the
    // quantized scan matrix (brute force, or HNSW per `vec:algorithm` -- the
    // graph reads the scan bytes) for the top-`rerankK` candidates;
    // (2) the rerank pass recomputes their distances EXACTLY on the fine
    // rerank matrix and keeps the top `k`. `maxDistance` filters on the FINE
    // distance only -- the coarse pass must not drop near-boundary candidates
    // by their quantized distance.
    const size_t rerankK = std::max(
        config.rerankK_.value_or(std::max<size_t>(10 * config.k_, 100)),
        config.k_);
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
    results = queryEntity.has_value()
                  ? vidx->searchExactByEntity(queryEntity.value(), config.k_,
                                              candidates, config.maxDistance_,
                                              checkInterrupt)
                  : vidx->searchExact(query, config.k_, candidates,
                                      config.maxDistance_, checkInterrupt);
  } else if (queryEntity.has_value()) {
    results =
        useHnsw ? vidx->searchHnswByEntity(queryEntity.value(), config.k_,
                                           config.maxDistance_, checkInterrupt)
                : vidx->searchExactByEntity(queryEntity.value(), config.k_,
                                            std::nullopt, config.maxDistance_,
                                            checkInterrupt);
  } else {
    results = useHnsw ? vidx->searchHnsw(query, config.k_, config.maxDistance_,
                                         checkInterrupt)
                      : vidx->searchExact(query, config.k_, std::nullopt,
                                          config.maxDistance_, checkInterrupt);
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
  // optional coarse `?coarseScore` (each occupying the next column iff
  // present).
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
}

// ____________________________________________________________________________
std::string VectorSearch::getDescriptor() const {
  return absl::StrCat("VectorSearch on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearch::getResultWidth() const {
  return 1 + static_cast<size_t>(config_.scoreVariable_.has_value()) +
         static_cast<size_t>(config_.coarseScoreVariable_.has_value());
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
  if (config_.rerankK_.has_value()) {
    absl::StrAppend(&key, " rerankK=", config_.rerankK_.value());
  }
  if (config_.maxDistance_.has_value()) {
    // Bit-exact so that two nearby values never share a key.
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
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
