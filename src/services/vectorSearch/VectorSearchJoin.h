// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHJOIN_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHJOIN_H

#include <memory>

#include "engine/MagicServicePlanning.h"
#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "services/vectorSearch/VectorSearchConfig.h"

// The "for each ?x, find the k most similar entities" form of vector search.
// For each row of the child subtree the operation looks up the query entity's
// vector and emits the k nearest entities of the index, carrying the child's
// columns through and adding `?result` (and optionally `?score`).
//
// The query entities can come from either side of the SERVICE clause:
//  * a nested `{ ... }` pattern inside the SERVICE that binds `<left>`
//    (the operation is then constructed complete), or
//  * the SURROUNDING query (mirroring `SpatialJoin`): the operation is planned
//    as an INCOMPLETE leaf implementing `IncompleteJoinOperation`, and the
//    join enumeration completes it with the subtree that binds `<left>`.
class VectorSearchJoin : public Operation, public IncompleteJoinOperation {
 private:
  qlever::vector::VectorSearchConfiguration config_;
  // The subtree binding the `<left>` variable; `nullptr` while incomplete.
  std::shared_ptr<QueryExecutionTree> child_;
  ColumnIndex leftCol_ = 0;  // column of the query variable in `child_`
  VariableToColumnMap variableColumns_;

 public:
  // Complete form (`child` binds the `<left>` variable) if `child` is
  // non-null; incomplete form otherwise (the planner adds the child later via
  // `addJoinChild`).
  VectorSearchJoin(QueryExecutionContext* qec,
                   qlever::vector::VectorSearchConfiguration config,
                   std::shared_ptr<QueryExecutionTree> child = nullptr);

  std::string getDescriptor() const override;
  size_t getResultWidth() const override;
  std::vector<ColumnIndex> resultSortedOn() const override { return {}; }
  bool knownEmptyResult() override {
    return config_.k_ == 0 || (child_ && child_->knownEmptyResult());
  }
  float getMultiplicity(size_t col) override;
  std::vector<QueryExecutionTree*> getChildren() override {
    return child_ ? std::vector{child_.get()}
                  : std::vector<QueryExecutionTree*>{};
  }
  size_t getCostEstimate() override;

  // `IncompleteJoinOperation` interface (see `engine/MagicServicePlanning.h`).
  bool isJoinConstructed() const override { return child_ != nullptr; }
  const Variable& joinVariable() const override {
    return config_.leftVariable_.value();
  }
  std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child) const override;

 private:
  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHJOIN_H
