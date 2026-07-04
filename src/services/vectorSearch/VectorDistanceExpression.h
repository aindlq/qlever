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
#include "services/vectorSearch/VectorSearchConfig.h"

namespace sparqlExpression {

// The `vec:distance*` SPARQL expression family: a per-row double distance from
// a fixed query point to the vector stored for `?entity` in the named vector
// index. Rows whose entity has no (live) vector evaluate to `UNDEF`. Combined
// with QLever's own `BIND`, `ORDER BY`, and `LIMIT`, this IS filtered top-k
// vector search -- no bespoke operator needed:
//
//   BIND(vec:distance("clip", ?e, "0.1,0.2,...") AS ?d)
//   FILTER(BOUND(?d)) ... ORDER BY ?d LIMIT k
//
// IMPORTANT: keep the `FILTER(BOUND(?d))` when the bound `?entity` set may
// contain entities WITHOUT a vector. `UNDEF` sorts BEFORE every real value in
// ascending `ORDER BY`, so without the filter a `LIMIT k` would return
// vectorless entities (with an `UNDEF` distance) first instead of the k
// nearest. The filter is unnecessary only when every candidate is known to
// have a vector.
//
// The query point is a CONSTANT, in one of four forms (the function IRI picks
// which): an explicit float-list literal or a constant entity IRI whose stored
// vector is used (`vec:distance`); free text (`vec:distanceText`); or an image
// URL/base64 (`vec:distanceImage`). Text and images are embedded ONCE, at query
// time, via the SAME endpoint the index was built with (looked up by the index
// name) -- so the query is guaranteed to use the model that produced the stored
// vectors, and no endpoint URL ever appears in the query. This lives entirely
// in the vector-search service folder and is wired into the parser via the
// generic `parsedQuery::SparqlFunctionRegistry`.
class VectorDistanceExpression : public SparqlExpression {
 public:
  using ImageQuery = qlever::vector::VectorSearchConfiguration::ImageQuery;

  // The query point is given by EXACTLY ONE of `queryVector`/`queryEntityIri`/
  // `queryText`/`queryImage`. `entity` is the operand expression evaluated per
  // row (typically a variable).
  VectorDistanceExpression(
      std::string indexName, Ptr entity,
      std::optional<std::vector<float>> queryVector,
      std::optional<ad_utility::triple_component::Iri> queryEntityIri,
      std::optional<std::string> queryText,
      std::optional<ImageQuery> queryImage);

  ExpressionResult evaluate(EvaluationContext* context) const override;
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override;

 private:
  ql::span<Ptr> childrenImpl() override;

  std::string indexName_;
  Ptr entity_;
  std::optional<std::vector<float>> queryVector_;
  std::optional<ad_utility::triple_component::Iri> queryEntityIri_;
  std::optional<std::string> queryText_;
  std::optional<ImageQuery> queryImage_;
  // The text/image query point embedded ONCE (per evaluation of this
  // expression object), so repeated per-block `evaluate` calls don't re-embed.
  mutable std::optional<std::vector<float>> embeddedQuery_;
};

// Factories used by the `SparqlFunctionRegistry` (arity 3: an index-name string
// literal, an operand expression for the entity, and the constant query point).
// `Text` reads a text literal; `Image` reads an image URL (IRI) or base64
// literal. Throw a user-facing error on a wrong arity or argument kind.
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args);
SparqlExpression::Ptr makeVectorDistanceTextExpression(
    std::vector<SparqlExpression::Ptr> args);
SparqlExpression::Ptr makeVectorDistanceImageExpression(
    std::vector<SparqlExpression::Ptr> args);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H
