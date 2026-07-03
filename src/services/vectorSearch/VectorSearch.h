// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCH_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCH_H

#include <memory>

#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "services/vectorSearch/VectorSearchConfig.h"

// A vector similarity search for a single query point (an explicit vector, the
// vector of a constant entity, or an embedded text/image), producing a
// `(?result[, ?score])` table of the k nearest entities.
//
// If a nested query pattern is given (the optional `candidates_` child that
// binds `?result`), the search is restricted to exactly those candidate
// entities via *exact* search -- the "small candidate set" optimisation (e.g.
// "among these 5000 green statues, the 10 most similar to <image>"). Without
// it, the whole index is searched (HNSW if available, else exact).
//
// (The "for each ?x, find similar" form is `VectorSearchJoin`.)
class VectorSearch : public Operation {
 private:
  qlever::vector::VectorSearchConfiguration config_;
  // Optional candidate-restriction child (binds `config_.resultVariable_`).
  std::shared_ptr<QueryExecutionTree> candidates_;
  ColumnIndex candidateCol_ = 0;  // column of resultVariable_ in `candidates_`
  VariableToColumnMap variableColumns_;

 public:
  VectorSearch(QueryExecutionContext* qec,
               qlever::vector::VectorSearchConfiguration config,
               std::shared_ptr<QueryExecutionTree> candidates = nullptr);

  std::string getDescriptor() const override;
  size_t getResultWidth() const override;
  std::vector<ColumnIndex> resultSortedOn() const override { return {}; }
  bool knownEmptyResult() override {
    return config_.k_ == 0 || (candidates_ && candidates_->knownEmptyResult());
  }
  float getMultiplicity(size_t col) override;
  std::vector<QueryExecutionTree*> getChildren() override {
    return candidates_ ? std::vector{candidates_.get()}
                       : std::vector<QueryExecutionTree*>{};
  }
  size_t getCostEstimate() override;

 private:
  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  // NOTE: results are cacheable for EVERY query form, including `queryText` and
  // `queryImage`. Their embedding is the expensive part (a round-trip to an
  // external endpoint), and the endpoint URL + model are fixed for the whole
  // run (they come from the index metadata), so the exact query text/image --
  // which is part of the cache key -- maps to a stable result within a run. The
  // result cache is in-memory and cleared on restart, so a later model change
  // cannot serve a stale result across it. (Hence no `canResultBeCachedImpl`
  // override; the base default of `true` applies.)
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCH_H
