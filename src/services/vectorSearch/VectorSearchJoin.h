// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#ifndef QLEVER_SRC_ENGINE_VECTORSEARCHJOIN_H
#define QLEVER_SRC_ENGINE_VECTORSEARCHJOIN_H

#include <memory>

#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "services/vectorSearch/VectorSearchConfig.h"

// The "for each ?x, find the k most similar entities" form of vector search.
// The query entities come from a nested query pattern (the operation's single
// child); for each child row the operation looks up the query entity's vector
// and emits the k nearest entities of the index, carrying the child's columns
// through and adding `?result` (and optionally `?score`).
//
// (A future refinement will take the query variable from *outside* the SERVICE
// via the planner's join machinery, and restrict the search space to a nested
// right pattern -- enabling the exact-over-small-candidate-set optimisation.)
class VectorSearchJoin : public Operation {
 private:
  qlever::vector::VectorSearchConfiguration config_;
  std::shared_ptr<QueryExecutionTree> child_;
  ColumnIndex leftCol_;  // column of the query variable in `child_`
  VariableToColumnMap variableColumns_;

 public:
  VectorSearchJoin(QueryExecutionContext* qec,
                   qlever::vector::VectorSearchConfiguration config,
                   std::shared_ptr<QueryExecutionTree> child);

  std::string getDescriptor() const override;
  size_t getResultWidth() const override;
  std::vector<ColumnIndex> resultSortedOn() const override { return {}; }
  bool knownEmptyResult() override {
    return config_.k_ == 0 || child_->knownEmptyResult();
  }
  float getMultiplicity(size_t col) override;
  std::vector<QueryExecutionTree*> getChildren() override {
    return {child_.get()};
  }
  size_t getCostEstimate() override;

 private:
  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_ENGINE_VECTORSEARCHJOIN_H
