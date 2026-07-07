// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMBERSCAN_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMBERSCAN_H

#include <memory>
#include <string>

#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"

// The members of a vector index as a single-column, already-sorted operand:
// `<.../vectorSearch/index/NAME> vec:hasMember ?e` binds `?e` to exactly the
// entities that have a (live) vector in the index. It is a leaf scan of the
// index's entity keys -- no vectors are materialised.
//
// The index physically stores its entities as a `ValueId`-sorted list (the
// `.rowmap`, live entries only, sorted by id bits), and for the persistent ids
// stored there the bit order equals QLever's `Id` comparison order. So the
// members are emitted as ONE column already sorted the way a merge join
// expects (`resultSortedOn() == {0}`): the planner merge-joins it with the rest
// of the query, replacing the `vec:distance -> UNDEF -> FILTER(BOUND(?d))`
// membership idiom with a cheap, exact join. Alone, it enumerates precisely the
// whole index.
class VectorMemberScan : public Operation {
 private:
  std::string indexName_;
  Variable entityVariable_;
  VariableToColumnMap variableColumns_;

 public:
  VectorMemberScan(QueryExecutionContext* qec, std::string indexName,
                   Variable entityVariable);

  std::string getDescriptor() const override;
  size_t getResultWidth() const override { return 1; }
  // The single output column is the entity id column, emitted in ascending
  // `ValueId` (== rowmap id-bits) order -- exactly a merge join's sort order.
  std::vector<ColumnIndex> resultSortedOn() const override {
    return {ColumnIndex{0}};
  }
  // Each member entity appears exactly once.
  float getMultiplicity(size_t) override { return 1.0f; }
  bool knownEmptyResult() override { return getSizeEstimateBeforeLimit() == 0; }
  std::vector<QueryExecutionTree*> getChildren() override { return {}; }
  size_t getCostEstimate() override;

 private:
  uint64_t getSizeEstimateBeforeLimit() override;
  std::string getCacheKeyImpl() const override;
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;
  std::unique_ptr<Operation> cloneImpl() const override;
};

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORMEMBERSCAN_H
