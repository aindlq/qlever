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

namespace qlever::vector {
class VectorIndex;
}  // namespace qlever::vector

// The `vec:candidates` (alias `vec:left`) forms of vector search. The
// operation is planned as an INCOMPLETE leaf implementing
// `IncompleteJoinOperation` (mirroring `SpatialJoin`); the join enumeration
// completes it with the subtree of the SURROUNDING query that binds
// `<candidates>`. Which form runs is decided by the config's query point and
// by whether the planner found such a subtree:
//  * FORM P (PRE-FILTER; query point + completed child): every bound
//    candidate is scored by the exact distance of its STORED vector to the
//    fixed query point -- the search is RESTRICTED to the bound set
//    (coarse-scan-then-rerank on a two-layer index). `?in == ?out` annotates
//    the child's rows in place (all scored candidates, or the top-`k` of the
//    bound set if `vec:k` was given); a distinct `?out` appends the top-k of
//    the bound set as a fresh column.
//  * FORM E (ENTITY-TO-ENTITY; no query point + completed child): for each
//    bound candidate, the k nearest entities of its OWN stored vector are
//    emitted into the DISTINCT `?result` column (plus optionally `?score`),
//    carrying the child's columns through.
//  * FORM W (WHOLE-INDEX; query point, NEVER completed -- `<candidates>` is
//    unbound): identical to omitting `<candidates>`; requires
//    `?in == ?out` and delegates to `computeWholeIndexSearch`.
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
  // The single missing join variable is `<left>`.
  bool isJoinConstructed() const override { return child_ != nullptr; }
  bool canBindJoinVariable(const Variable& var) const override {
    return var == config_.leftVariable_.value();
  }
  std::string multipleJoinVariablesError() const override;
  std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child,
      const Variable& var) const override;

 private:
  // True iff `<result>` names the same variable as `<candidates>` (the FORM P
  // annotate-in-place spelling; also how FORM W is spelled with candidates).
  bool annotatesCandidatesInPlace() const {
    return config_.leftVariable_ == config_.resultVariable_;
  }

  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  Result computeResult(bool requestLaziness) override;
  // FORM P: score the child's bound candidates (and ONLY those) against the
  // fixed query point, appending the result rows to `result`.
  void computePreFilterRows(const qlever::vector::VectorIndex& vidx,
                            const IdTable& childTable, IdTable* result);
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHJOIN_H
