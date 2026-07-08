// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCH_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCH_H

#include <memory>
#include <string>

#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "services/vectorSearch/VectorSearchConfig.h"

namespace qlever::vector {

class VectorIndex;

// Append the query-point fields of `config` to a cache key, each value
// length-prefixed / bit-exact so distinct query points never share a key.
// Shared by `VectorSearch` and `VectorSearchJoin`.
void appendQueryPointToCacheKey(std::string* key,
                                const VectorSearchConfiguration& config);

// Throw a clear user-facing error when `vec:cslsThreshold` targets an index
// that was not built with `csls: true`. Shared by the FORM W and FORM P
// paths.
void validateCslsIsAvailable(const VectorSearchConfiguration& config,
                             const VectorIndex& vidx);

// The WHOLE-INDEX top-k search of `config` (FORM W): resolve the query point
// against the index `config.indexName_` and return the k nearest entities as
// a `(?result[, ?score][, ?coarseScore])` table in that column order
// (coarse-scan-then-rerank on a two-layer index). Shared by the `VectorSearch`
// operation and by a never-completed `VectorSearchJoin` leaf (FORM W spelled
// as `vec:candidates ?out` with `?out` unbound by the surrounding query).
IdTable computeWholeIndexSearch(
    const VectorSearchConfiguration& config, QueryExecutionContext* qec,
    const ad_utility::SharedCancellationHandle& handle);

}  // namespace qlever::vector

// A WHOLE-INDEX vector similarity search for a single query point (an explicit
// vector, the vector of a constant entity, or an embedded text/image),
// producing a `(?result[, ?score][, ?coarseScore])` table of the k nearest
// entities of the index (HNSW if available, else exact). This is the "search"
// surface.
//
// On a TWO-LAYER index (built with `rerank`) the search runs coarse-scan-then-
// rerank: the top-`rerankK` candidates come off the quantized scan matrix
// (brute force or HNSW), their distances are then recomputed EXACTLY on the
// fine rerank matrix, and the top `k` by fine distance are returned.
// `vec:bindScore` binds the fine distance (== the exact `vec:distance` of the
// entity); `vec:bindCoarseScore` optionally binds the coarse scan distance
// alongside it, so `ABS(?d - ?dc)` exposes the quantization error. EXCEPTION:
// on a `binary` scan layer the coarse distance is the integer HAMMING
// distance (0..dim differing sign bits) -- a ranking proxy on a different
// scale than the fine cosine distance, deliberately NOT reconciled with it
// (`ABS(?d - ?dc)` is meaningless there, unlike i8).
//
// (The "for each ?x, find similar" form is `VectorSearchJoin`; ranking an
// existing candidate set is the `vec:distance` function + `ORDER BY`/`LIMIT`.)
class VectorSearch : public Operation {
 private:
  qlever::vector::VectorSearchConfiguration config_;
  VariableToColumnMap variableColumns_;

 public:
  VectorSearch(QueryExecutionContext* qec,
               qlever::vector::VectorSearchConfiguration config);

  std::string getDescriptor() const override;
  size_t getResultWidth() const override;
  std::vector<ColumnIndex> resultSortedOn() const override { return {}; }
  bool knownEmptyResult() override { return config_.k_ == 0; }
  float getMultiplicity(size_t col) override;
  std::vector<QueryExecutionTree*> getChildren() override { return {}; }
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
