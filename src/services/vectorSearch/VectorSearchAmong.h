// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHAMONG_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHAMONG_H

#include <memory>

#include "engine/MagicServicePlanning.h"
#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "services/vectorSearch/VectorSearchConfig.h"

// The `vec:among ?s` form: an outer-bound candidate restriction. The candidate
// set `?s` is bound by the SURROUNDING query (patterns OUTSIDE the SERVICE, no
// nesting); the operation returns the top-k of those candidates by distance to
// a fixed query point (an explicit vector, a constant entity, or an embedded
// text/image). The result variable IS `?s` (the result is a subset of the
// candidates), and the child's other columns are carried through, plus an
// optional `?score`.
//
// It is planned as an INCOMPLETE `IncompleteJoinOperation` (mirroring the
// outer-bound `vec:left` form of `VectorSearchJoin`): the join enumeration
// completes it with the subtree that binds `?s`. Unlike `VectorSearchJoin`
// (which produces the k neighbours OF each child row, a fan-out), this REDUCES
// the child to its globally k nearest rows.
class VectorSearchAmong : public Operation, public IncompleteJoinOperation {
 private:
  qlever::vector::VectorSearchConfiguration config_;
  // The subtree binding the candidate/result variable; `nullptr` while
  // incomplete.
  std::shared_ptr<QueryExecutionTree> child_;
  ColumnIndex amongCol_ = 0;  // column of `resultVariable_` in `child_`
  VariableToColumnMap variableColumns_;

 public:
  // Complete form (`child` binds `config.resultVariable_`) if `child` is
  // non-null; incomplete form otherwise (the planner adds the child later via
  // `addJoinChild`).
  VectorSearchAmong(QueryExecutionContext* qec,
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
  // The single missing join variable is the candidate/result variable.
  bool isJoinConstructed() const override { return child_ != nullptr; }
  bool canBindJoinVariable(const Variable& var) const override {
    return var == config_.resultVariable_;
  }
  std::string multipleJoinVariablesError() const override;
  std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child,
      const Variable& var) const override;

 private:
  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHAMONG_H
