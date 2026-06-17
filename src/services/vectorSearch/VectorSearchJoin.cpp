// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorSearchJoin.h"

#include <absl/strings/str_cat.h>

#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"

VectorSearchJoin::VectorSearchJoin(
    QueryExecutionContext* qec,
    qlever::vector::VectorSearchConfiguration config,
    std::shared_ptr<QueryExecutionTree> child)
    : Operation{qec}, config_{std::move(config)}, child_{std::move(child)} {
  AD_CONTRACT_CHECK(config_.leftVariable_.has_value(),
                    "VectorSearchJoin requires a query (`<left>`) variable.");
  const auto& childCols = child_->getVariableColumns();
  auto it = childCols.find(config_.leftVariable_.value());
  AD_CONTRACT_CHECK(it != childCols.end(),
                    "The vector-search `<left>` variable ",
                    config_.leftVariable_.value().name(),
                    " is not bound by the nested query pattern.");
  leftCol_ = it->second.columnIndex_;

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

std::string VectorSearchJoin::getDescriptor() const {
  return absl::StrCat("VectorSearchJoin on index '", config_.indexName_,
                      "', k=", config_.k_);
}

size_t VectorSearchJoin::getResultWidth() const {
  return child_->getResultWidth() + (config_.scoreVariable_.has_value() ? 2 : 1);
}

float VectorSearchJoin::getMultiplicity(size_t col) {
  // Passthrough columns inherit the child's multiplicity (scaled by the k-fold
  // expansion); the synthetic result/score columns are treated as ~unique.
  size_t childWidth = child_->getResultWidth();
  if (col < childWidth) {
    return child_->getMultiplicity(col) * static_cast<float>(config_.k_);
  }
  return 1.0f;
}

uint64_t VectorSearchJoin::getSizeEstimateBeforeLimit() {
  return child_->getSizeEstimate() * config_.k_;
}

size_t VectorSearchJoin::getCostEstimate() {
  // Roughly: one index probe per child row, plus the child's own cost.
  return child_->getCostEstimate() + child_->getSizeEstimate() * config_.k_;
}

VariableToColumnMap VectorSearchJoin::computeVariableToColumnMap() const {
  return variableColumns_;
}

std::string VectorSearchJoin::getCacheKeyImpl() const {
  return absl::StrCat("VECTOR_SEARCH_JOIN index=", config_.indexName_,
                      " k=", config_.k_,
                      " algo=", static_cast<int>(config_.algorithm_),
                      " leftCol=", leftCol_,
                      " score=", config_.scoreVariable_.has_value(), " {",
                      child_->getCacheKey(), "}");
}

std::unique_ptr<Operation> VectorSearchJoin::cloneImpl() const {
  return std::make_unique<VectorSearchJoin>(getExecutionContext(), config_,
                                            child_->clone());
}

Result VectorSearchJoin::computeResult([[maybe_unused]] bool requestLaziness) {
  const Index& index = getExecutionContext()->getIndex();
  const qlever::vector::VectorIndex* vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  AD_CONTRACT_CHECK(vidx != nullptr,
                    "There is no loaded vector index named '",
                    config_.indexName_, "'.");
  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  AD_CONTRACT_CHECK(!(config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()),
                    "Vector search requested the HNSW algorithm, but index '",
                    config_.indexName_, "' has no HNSW structure.");
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  bool withScore = config_.scoreVariable_.has_value();

  std::shared_ptr<const Result> childRes = child_->getResult();
  const IdTable& childTable = childRes->idTable();
  const size_t childWidth = childTable.numColumns();

  IdTable result{getResultWidth(), getExecutionContext()->getAllocator()};
  for (size_t row = 0; row < childTable.numRows(); ++row) {
    checkCancellation();
    Id leftId = childTable(row, leftCol_);
    auto queryVec = vidx->getVector(leftId);
    if (!queryVec.has_value()) {
      continue;  // query entity has no vector -> contributes no result rows
    }
    std::vector<qlever::vector::ScoredEntity> hits =
        useHnsw ? vidx->searchHnsw(queryVec.value(), config_.k_)
                : vidx->searchExact(queryVec.value(), config_.k_);
    for (const auto& hit : hits) {
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
  return {std::move(result), resultSortedOn(),
          childRes->getCopyOfLocalVocab()};
}
