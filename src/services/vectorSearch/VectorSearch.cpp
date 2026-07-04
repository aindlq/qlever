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

namespace {
// Append a string to a cache key unambiguously (length-prefixed, so crafted
// values cannot collide with other parts of the key).
void appendToKey(std::string* key, std::string_view field,
                 std::string_view value) {
  absl::StrAppend(key, " ", field, "=", value.size(), ":", value);
}
}  // namespace

// ____________________________________________________________________________
VectorSearch::VectorSearch(QueryExecutionContext* qec,
                           qlever::vector::VectorSearchConfiguration config)
    : Operation{qec}, config_{std::move(config)} {
  variableColumns_[config_.resultVariable_] =
      makeAlwaysDefinedColumn(ColumnIndex{0});
  if (config_.scoreVariable_.has_value()) {
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{1});
  }
}

// ____________________________________________________________________________
std::string VectorSearch::getDescriptor() const {
  return absl::StrCat("VectorSearch on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearch::getResultWidth() const {
  return config_.scoreVariable_.has_value() ? 2 : 1;
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
  std::string key = absl::StrCat("VECTOR_SEARCH index=", config_.indexName_,
                                 " k=", config_.k_,
                                 " algo=", static_cast<int>(config_.algorithm_),
                                 " score=", config_.scoreVariable_.has_value());
  if (config_.maxDistance_.has_value()) {
    // Bit-exact so that two nearby values never share a key.
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  if (config_.queryEntityIri_.has_value()) {
    appendToKey(&key, "entity", config_.queryEntityIri_.value());
  }
  if (config_.queryVector_.has_value()) {
    absl::StrAppend(&key, " vec=[");
    for (float f : config_.queryVector_.value()) {
      // Bit-exact: `absl::StrCat(float)` would round to six significant
      // digits, making distinct query vectors collide on one key.
      absl::StrAppend(&key, absl::Hex(absl::bit_cast<uint32_t>(f)), ",");
    }
    absl::StrAppend(&key, "]");
  }
  if (config_.queryText_.has_value()) {
    appendToKey(&key, "text", config_.queryText_.value());
  }
  if (config_.queryImage_.has_value()) {
    absl::StrAppend(&key, " imageKind=",
                    static_cast<int>(config_.queryImage_.value().kind_));
    appendToKey(&key, "image", config_.queryImage_.value().value_);
  }
  return key;
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorSearch::cloneImpl() const {
  return std::make_unique<VectorSearch>(getExecutionContext(), config_);
}

// ____________________________________________________________________________
Result VectorSearch::computeResult([[maybe_unused]] bool requestLaziness) {
  const Index& index = getExecutionContext()->getIndex();
  std::shared_ptr<const qlever::vector::VectorIndex> vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  if (!vidx) {
    throw std::runtime_error{absl::StrCat(
        "There is no loaded vector index named '", config_.indexName_,
        "'. Was the index built with `--service-index`?")};
  }

  IdTable idTable{getResultWidth(), getExecutionContext()->getAllocator()};

  // Resolve the query point (shared with the `vec:distance` function): an
  // explicit or embedded vector goes into `query`; a constant entity
  // (`vec:query <iri>`) is searched by its STORED vector directly (no
  // decode/re-encode through f32), tracked in `queryEntity`; an
  // unknown/vectorless entity -> empty result.
  qlever::vector::QueryPoint queryPoint = qlever::vector::resolveQueryPoint(
      config_, *vidx, index.getImpl(), cancellationHandle_);
  if (std::holds_alternative<std::monostate>(queryPoint)) {
    return {std::move(idTable), resultSortedOn(), LocalVocab{}};
  }
  std::vector<float> query;
  std::optional<Id> queryEntity;
  if (std::holds_alternative<Id>(queryPoint)) {
    queryEntity = std::get<Id>(queryPoint);
  } else {
    query = std::move(std::get<std::vector<float>>(queryPoint));
  }

  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  if (config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()) {
    throw std::runtime_error{
        absl::StrCat("Vector search requested the HNSW algorithm, but index '",
                     config_.indexName_, "' has no HNSW structure.")};
  }
  auto checkInterrupt = [this]() { checkCancellation(); };

  // Whole-index search (by the stored vector for the entity form -- no
  // decode/re-encode through f32).
  std::vector<qlever::vector::ScoredEntity> results;
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  if (queryEntity.has_value()) {
    results =
        useHnsw ? vidx->searchHnswByEntity(queryEntity.value(), config_.k_,
                                           config_.maxDistance_, checkInterrupt)
                : vidx->searchExactByEntity(queryEntity.value(), config_.k_,
                                            std::nullopt, config_.maxDistance_,
                                            checkInterrupt);
  } else {
    results = useHnsw ? vidx->searchHnsw(query, config_.k_,
                                         config_.maxDistance_, checkInterrupt)
                      : vidx->searchExact(query, config_.k_, std::nullopt,
                                          config_.maxDistance_, checkInterrupt);
  }

  idTable.resize(results.size());
  bool withScore = config_.scoreVariable_.has_value();
  for (size_t i = 0; i < results.size(); ++i) {
    idTable(i, 0) = results[i].entity_;
    if (withScore) {
      idTable(i, 1) = Id::makeFromDouble(results[i].distance_);
    }
  }
  return {std::move(idTable), resultSortedOn(), LocalVocab{}};
}
