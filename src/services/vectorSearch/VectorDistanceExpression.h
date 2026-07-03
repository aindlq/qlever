// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H

#include <optional>
#include <string>
#include <vector>

#include "engine/sparqlExpressions/SparqlExpression.h"
#include "rdfTypes/Iri.h"

namespace sparqlExpression {

// The `vec:distance("index", ?entity, <query>)` SPARQL expression: a per-row
// double distance from a fixed query point to the vector stored for `?entity`
// in the named vector index. Rows whose entity has no (live) vector evaluate to
// `UNDEF`. Combined with QLever's own `BIND`, `ORDER BY`, and `LIMIT`, this IS
// filtered top-k vector search -- no bespoke operator needed:
//
//   BIND(vec:distance("clip", ?e, "0.1,0.2,...") AS ?d) ... ORDER BY ?d LIMIT k
//
// The query point is a constant: either an explicit comma-separated float
// literal, or a constant entity IRI whose stored vector is used as the query.
// (The index name is likewise a constant string literal.) The lives entirely in
// the vector-search service folder and is wired into the parser via the generic
// `parsedQuery::SparqlFunctionRegistry`.
class VectorDistanceExpression : public SparqlExpression {
 public:
  // The query point is given EITHER as an explicit vector OR as a constant
  // entity IRI (exactly one of `queryVector`/`queryEntityIri` is set). `entity`
  // is the operand expression evaluated per row (typically a variable).
  VectorDistanceExpression(
      std::string indexName, Ptr entity,
      std::optional<std::vector<float>> queryVector,
      std::optional<ad_utility::triple_component::Iri> queryEntityIri);

  ExpressionResult evaluate(EvaluationContext* context) const override;
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override;

 private:
  ql::span<Ptr> childrenImpl() override;

  std::string indexName_;
  Ptr entity_;
  std::optional<std::vector<float>> queryVector_;
  std::optional<ad_utility::triple_component::Iri> queryEntityIri_;
};

// Factory used by the `SparqlFunctionRegistry`: validate the three parsed
// arguments of `vec:distance(...)` -- a constant string literal (index name),
// an operand expression (the entity), and a constant query (a float-list string
// literal or an entity IRI) -- and build a `VectorDistanceExpression`. Throws a
// user-facing error on a wrong arity or argument kind.
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H
