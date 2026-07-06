// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORVECTOREXPRESSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORVECTOREXPRESSION_H

#include <string>
#include <vector>

#include "engine/sparqlExpressions/SparqlExpression.h"

namespace sparqlExpression {

// The `vec:vector(<.../index/NAME>, entity)` SPARQL expression: fetch the
// STORED vector of `entity` from the named vector index, decoded to f32, as a
// typed query-vector literal `"f0,f1,..."^^<.../vec/MODEL/PRECISION>` (the
// index's embedding model and storage precision; see
// `VEC_QUERY_DATATYPE_PREFIX`). An entity without a (live) vector in the index
// -- or anything that is not an entity at all -- evaluates to UNDEF for that
// row.
//
// This is the bridge for CROSS-INDEX distances: the datatype carries the
// source index's embedding space, and `vec:distance` VALIDATES it against the
// index the distance is computed in (same model, same precision, same
// dimension), so
//
//   BIND(vec:distance(vidx:artwork, ?a, vec:vector(vidx:photo, ?p)) AS ?d)
//
// computes iff the two indices share a space and is UNDEF otherwise. Within
// ONE index, entity<->entity distance does NOT need this function --
// `vec:distance(vidx:emb, ?a, ?b)` looks up both stored vectors internally
// without materializing either. A constant entity resolves ONCE (the constant
// sub-expression yields a constant result); it needs no memoization beyond
// that, because the lookup is a cheap rowmap access, unlike `vec:embed`'s
// endpoint round trip.
class VectorVectorExpression : public SparqlExpression {
 public:
  VectorVectorExpression(std::string indexName, Ptr entity);

  ExpressionResult evaluate(EvaluationContext* context) const override;
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override;

 private:
  ql::span<Ptr> childrenImpl() override;

  std::string indexName_;
  Ptr entity_;
};

// Factory used by the `SparqlFunctionRegistry` (arity 2: the index IRI
// `<.../vectorSearch/index/NAME>` and the entity expression). Throws a
// user-facing error on a wrong arity or a malformed index argument.
SparqlExpression::Ptr makeVectorVectorExpression(
    std::vector<SparqlExpression::Ptr> args);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORVECTOREXPRESSION_H
