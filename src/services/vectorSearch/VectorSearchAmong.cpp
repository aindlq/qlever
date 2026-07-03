// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSearchAmong.h"

#include <absl/base/casts.h>
#include <absl/strings/str_cat.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorQueryPoint.h"

// ____________________________________________________________________________
VectorSearchAmong::VectorSearchAmong(
    QueryExecutionContext* qec,
    qlever::vector::VectorSearchConfiguration config,
    std::shared_ptr<QueryExecutionTree> child)
    : Operation{qec}, config_{std::move(config)}, child_{std::move(child)} {
  AD_CONTRACT_CHECK(config_.amongVariable_.has_value(),
                    "VectorSearchAmong requires an `<among>` variable.");
  if (!child_) {
    // INCOMPLETE form: the subtree binding the candidate/result variable comes
    // from the surrounding query. Expose the output variables as
    // possibly-undefined columns so the join enumeration can find the
    // connection (see `IncompleteJoinOperation`); `addJoinChild` builds the
    // completed operation.
    variableColumns_[config_.resultVariable_] =
        makePossiblyUndefinedColumn(ColumnIndex{0});
    if (config_.scoreVariable_.has_value()) {
      variableColumns_[config_.scoreVariable_.value()] =
          makePossiblyUndefinedColumn(ColumnIndex{1});
    }
    return;
  }
  const auto& childCols = child_->getVariableColumns();
  auto it = childCols.find(config_.resultVariable_);
  if (it == childCols.end()) {
    throw std::runtime_error{absl::StrCat(
        "The vector-search `<among>` variable ", config_.resultVariable_.name(),
        " is not bound by the surrounding query, so there is no candidate set "
        "to rank.")};
  }
  amongCol_ = it->second.columnIndex_;

  // The result variable is already one of the child's columns (it is the
  // candidate set). Carry ALL child columns through; the top-k rows are a
  // subset of them. Only `?score` is a new column.
  variableColumns_ = childCols;
  size_t childWidth = child_->getResultWidth();
  if (config_.scoreVariable_.has_value()) {
    if (childCols.contains(config_.scoreVariable_.value())) {
      throw std::runtime_error{
          absl::StrCat("The vector-search `<bindScore>` variable ",
                       config_.scoreVariable_.value().name(),
                       " must not be bound by the surrounding query.")};
    }
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{childWidth});
  }
}

// ____________________________________________________________________________
std::string VectorSearchAmong::multipleJoinVariablesError() const {
  return absl::StrCat(
      "A vector search whose `<among>` variable ",
      config_.resultVariable_.name(),
      " comes from the surrounding query must be connected to the rest of the "
      "query through that variable only; it also shares another of its "
      "variables. Rename the shared variable, or bind the extra connection in "
      "a separate pattern.");
}

// ____________________________________________________________________________
std::shared_ptr<Operation> VectorSearchAmong::addJoinChild(
    std::shared_ptr<QueryExecutionTree> child, const Variable& var) const {
  AD_CONTRACT_CHECK(var == config_.resultVariable_);
  auto completed = std::make_shared<VectorSearchAmong>(
      getExecutionContext(), config_, std::move(child));
  for (const auto& warning : *getWarnings().rlock()) {
    completed->addWarning(warning);
  }
  return completed;
}

// ____________________________________________________________________________
std::string VectorSearchAmong::getDescriptor() const {
  return absl::StrCat("VectorSearchAmong on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearchAmong::getResultWidth() const {
  size_t ownColumns = config_.scoreVariable_.has_value() ? 1 : 0;
  if (!child_) {
    // Incomplete: `<among>`[, `<bindScore>`].
    return 1 + ownColumns;
  }
  return child_->getResultWidth() + ownColumns;
}

// ____________________________________________________________________________
float VectorSearchAmong::getMultiplicity(size_t col) {
  if (!child_) {
    return 1.0f;
  }
  // The result is a subset of the child rows, so passthrough columns keep the
  // child's multiplicity; the synthetic score column is ~unique.
  size_t childWidth = child_->getResultWidth();
  if (col < childWidth) {
    return child_->getMultiplicity(col);
  }
  return 1.0f;
}

// ____________________________________________________________________________
uint64_t VectorSearchAmong::getSizeEstimateBeforeLimit() {
  if (!child_) {
    return config_.k_;
  }
  return std::min<uint64_t>(config_.k_, child_->getSizeEstimate());
}

// ____________________________________________________________________________
size_t VectorSearchAmong::getCostEstimate() {
  if (!child_) {
    return config_.k_;
  }
  // The child's own cost plus one (cheap) distance computation per child row.
  return child_->getCostEstimate() + child_->getSizeEstimate();
}

// ____________________________________________________________________________
VariableToColumnMap VectorSearchAmong::computeVariableToColumnMap() const {
  return variableColumns_;
}

// ____________________________________________________________________________
std::string VectorSearchAmong::getCacheKeyImpl() const {
  std::string key = absl::StrCat(
      "VECTOR_SEARCH_AMONG index=", config_.indexName_, " k=", config_.k_,
      " amongCol=", amongCol_, " score=", config_.scoreVariable_.has_value());
  // The query point participates in the result, so it must be part of the key.
  if (config_.queryEntityIri_.has_value()) {
    absl::StrAppend(&key, " entity=", config_.queryEntityIri_.value());
  }
  if (config_.queryVector_.has_value()) {
    absl::StrAppend(&key, " vec=[");
    for (float f : config_.queryVector_.value()) {
      absl::StrAppend(&key, absl::Hex(absl::bit_cast<uint32_t>(f)), ",");
    }
    absl::StrAppend(&key, "]");
  }
  if (config_.queryText_.has_value()) {
    absl::StrAppend(&key, " text=", config_.queryText_.value().size(), ":",
                    config_.queryText_.value());
  }
  if (config_.queryImage_.has_value()) {
    absl::StrAppend(&key, " imageKind=",
                    static_cast<int>(config_.queryImage_.value().kind_),
                    " image=", config_.queryImage_.value().value_.size(), ":",
                    config_.queryImage_.value().value_);
  }
  if (config_.maxDistance_.has_value()) {
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  if (child_) {
    absl::StrAppend(&key, " {", child_->getCacheKey(), "}");
  } else {
    absl::StrAppend(&key, " INCOMPLETE");
  }
  return key;
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorSearchAmong::cloneImpl() const {
  return std::make_unique<VectorSearchAmong>(
      getExecutionContext(), config_, child_ ? child_->clone() : nullptr);
}

// ____________________________________________________________________________
Result VectorSearchAmong::computeResult([[maybe_unused]] bool requestLaziness) {
  if (!child_) {
    throw std::runtime_error{absl::StrCat(
        "The vector-search `<among>` variable ", config_.resultVariable_.name(),
        " is not bound anywhere: bind it in the surrounding query so it can be "
        "ranked (it is the candidate set of the `<among>` form).")};
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
  const size_t childWidth = childTable.numColumns();
  bool withScore = config_.scoreVariable_.has_value();

  // Resolve the (fixed) query point and build a distance functor for it.
  qlever::vector::QueryPoint queryPoint = qlever::vector::resolveQueryPoint(
      config_, *vidx, index.getImpl(), cancellationHandle_);
  if (std::holds_alternative<std::monostate>(queryPoint)) {
    // Unknown / vectorless query entity -> empty result.
    IdTable empty{getResultWidth(), getExecutionContext()->getAllocator()};
    return {std::move(empty), resultSortedOn(),
            childRes->getCopyOfLocalVocab()};
  }
  std::optional<qlever::vector::VectorIndex::DistanceComputer> computer;
  if (std::holds_alternative<Id>(queryPoint)) {
    computer = vidx->makeDistanceComputerByEntity(std::get<Id>(queryPoint));
    // `resolveQueryPoint` only returns an `Id` that has a live vector.
    AD_CORRECTNESS_CHECK(computer.has_value());
  } else {
    computer =
        vidx->makeDistanceComputer(std::get<std::vector<float>>(queryPoint));
  }
  const auto& computeDistance = computer.value();

  // Bounded top-k of the child rows by distance (a max-heap of size k, keyed by
  // distance). A candidate whose entity has no vector (NaN) or that exceeds the
  // optional distance cutoff is dropped.
  struct Cand {
    float dist_;
    size_t row_;
  };
  auto worseFirst = [](const Cand& a, const Cand& b) {
    return a.dist_ < b.dist_;  // max-heap on distance (the largest at the top)
  };
  std::vector<Cand> top;
  const size_t k = config_.k_;
  for (size_t row = 0; row < childTable.numRows(); ++row) {
    if (row % 1024 == 0) {
      checkCancellation();
    }
    if (k == 0) {
      break;
    }
    float dist = computeDistance(childTable(row, amongCol_));
    if (std::isnan(dist) ||
        (config_.maxDistance_.has_value() && dist > config_.maxDistance_)) {
      continue;
    }
    if (top.size() < k) {
      top.push_back(Cand{dist, row});
      std::push_heap(top.begin(), top.end(), worseFirst);
    } else if (dist < top.front().dist_) {
      std::pop_heap(top.begin(), top.end(), worseFirst);
      top.back() = Cand{dist, row};
      std::push_heap(top.begin(), top.end(), worseFirst);
    }
  }
  // `sort_heap` on a max-heap yields ascending order (nearest first).
  std::sort_heap(top.begin(), top.end(), worseFirst);

  IdTable result{getResultWidth(), getExecutionContext()->getAllocator()};
  result.resize(top.size());
  for (size_t i = 0; i < top.size(); ++i) {
    for (size_t c = 0; c < childWidth; ++c) {
      result(i, c) = childTable(top[i].row_, c);
    }
    if (withScore) {
      result(i, childWidth) = Id::makeFromDouble(top[i].dist_);
    }
  }
  return {std::move(result), resultSortedOn(), childRes->getCopyOfLocalVocab()};
}
