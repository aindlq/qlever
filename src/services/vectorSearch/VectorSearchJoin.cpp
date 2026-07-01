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
#include "util/HashMap.h"

// ____________________________________________________________________________
VectorSearchJoin::VectorSearchJoin(
    QueryExecutionContext* qec,
    qlever::vector::VectorSearchConfiguration config,
    std::shared_ptr<QueryExecutionTree> child)
    : Operation{qec}, config_{std::move(config)}, child_{std::move(child)} {
  AD_CONTRACT_CHECK(config_.leftVariable_.has_value(),
                    "VectorSearchJoin requires a query (`<left>`) variable.");
  if (!child_) {
    // INCOMPLETE form: the subtree binding `<left>` comes from the
    // surrounding query. Expose the variables as possibly-undefined columns
    // so that the join enumeration can find the connection (see
    // `IncompleteJoinOperation`); `addJoinChild` builds the completed
    // operation.
    variableColumns_[config_.leftVariable_.value()] =
        makePossiblyUndefinedColumn(ColumnIndex{0});
    variableColumns_[config_.resultVariable_] =
        makePossiblyUndefinedColumn(ColumnIndex{1});
    if (config_.scoreVariable_.has_value()) {
      variableColumns_[config_.scoreVariable_.value()] =
          makePossiblyUndefinedColumn(ColumnIndex{2});
    }
    return;
  }
  const auto& childCols = child_->getVariableColumns();
  auto it = childCols.find(config_.leftVariable_.value());
  if (it == childCols.end()) {
    throw std::runtime_error{
        absl::StrCat("The vector-search `<left>` variable ",
                     config_.leftVariable_.value().name(),
                     " is not bound by the nested query pattern.")};
  }
  leftCol_ = it->second.columnIndex_;

  // The `?result`/`?score` columns are appended to the child's columns; if the
  // nested pattern also binds one of them, silently overwriting the mapping
  // would relabel a child column and return wrong results -- reject instead.
  auto checkNotBoundByChild = [&childCols](const Variable& var,
                                           std::string_view parameter) {
    if (childCols.contains(var)) {
      throw std::runtime_error{absl::StrCat(
          "The vector-search `<", parameter, ">` variable ", var.name(),
          " must not be bound by the nested query pattern. (Restricting the "
          "search space of the `<left>` form is not supported yet.)")};
    }
  };
  checkNotBoundByChild(config_.resultVariable_, "result");
  if (config_.scoreVariable_.has_value()) {
    checkNotBoundByChild(config_.scoreVariable_.value(), "bindScore");
  }

  // Output columns: all child columns, then `?result`, then optional `?score`.
  variableColumns_ = childCols;
  size_t childWidth = child_->getResultWidth();
  variableColumns_[config_.resultVariable_] =
      makeAlwaysDefinedColumn(ColumnIndex{childWidth});
  if (config_.scoreVariable_.has_value()) {
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{childWidth + 1});
  }
}

// ____________________________________________________________________________
std::shared_ptr<Operation> VectorSearchJoin::addJoinChild(
    std::shared_ptr<QueryExecutionTree> child) const {
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
  size_t ownColumns = config_.scoreVariable_.has_value() ? 2 : 1;
  if (!child_) {
    // Incomplete: `<left>`, `<result>`[, `<bindScore>`].
    return 1 + ownColumns;
  }
  return child_->getResultWidth() + ownColumns;
}

// ____________________________________________________________________________
float VectorSearchJoin::getMultiplicity(size_t col) {
  if (!child_) {
    return 1.0f;
  }
  // Passthrough columns inherit the child's multiplicity (scaled by the k-fold
  // expansion); the synthetic result/score columns are treated as ~unique.
  size_t childWidth = child_->getResultWidth();
  if (col < childWidth) {
    return child_->getMultiplicity(col) * static_cast<float>(config_.k_);
  }
  return 1.0f;
}

// ____________________________________________________________________________
uint64_t VectorSearchJoin::getSizeEstimateBeforeLimit() {
  return child_ ? child_->getSizeEstimate() * config_.k_ : config_.k_;
}

// ____________________________________________________________________________
size_t VectorSearchJoin::getCostEstimate() {
  // The child's own cost plus one index probe per child row. A probe is ~log(N)
  // with HNSW, but a full scan of the N vectors without it.
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
      " algo=", static_cast<int>(config_.algorithm_), " leftCol=", leftCol_,
      " score=", config_.scoreVariable_.has_value());
  if (config_.maxDistance_.has_value()) {
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  if (child_) {
    absl::StrAppend(&key, " {", child_->getCacheKey(), "}");
  } else {
    // Incomplete operations are never executed or cached; the marker only
    // keeps the key well-defined during planning.
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
    throw std::runtime_error{absl::StrCat(
        "The vector-search `<left>` variable ",
        config_.leftVariable_.value().name(),
        " is not bound anywhere: bind it in the surrounding query (it is "
        "then joined with the vector search) or in a nested `{ ... }` "
        "pattern inside the SERVICE clause.")};
  }
  const Index& index = getExecutionContext()->getIndex();
  std::shared_ptr<const qlever::vector::VectorIndex> vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  if (!vidx) {
    throw std::runtime_error{absl::StrCat(
        "There is no loaded vector index named '", config_.indexName_,
        "'. Was the index built with `--service-index`?")};
  }
  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  if (config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()) {
    throw std::runtime_error{
        absl::StrCat("Vector search requested the HNSW algorithm, but index '",
                     config_.indexName_, "' has no HNSW structure.")};
  }
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  bool withScore = config_.scoreVariable_.has_value();
  auto checkInterrupt = [this]() { checkCancellation(); };

  std::shared_ptr<const Result> childRes = child_->getResult();
  const IdTable& childTable = childRes->idTable();
  const size_t childWidth = childTable.numColumns();

  // Children are often sorted (or at least clustered) by the join column, so
  // memoize the search results per distinct query entity -- without this, a
  // child with many duplicate `<left>` values re-runs identical searches.
  ad_utility::HashMap<uint64_t, std::vector<qlever::vector::ScoredEntity>>
      hitsByEntity;

  IdTable result{getResultWidth(), getExecutionContext()->getAllocator()};
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
