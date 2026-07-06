// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTOREMBEDEXPRESSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTOREMBEDEXPRESSION_H

#include <string>
#include <vector>

#include "engine/sparqlExpressions/SparqlExpression.h"
#include "util/HashMap.h"

namespace sparqlExpression {

// The `vec:embed(<.../index/NAME>, input)` SPARQL expression: embed `input`
// via the embedding endpoint the named vector index was built with (its
// `embeddingUrl`/`embeddingModel` build-spec keys), so a query is guaranteed
// to be embedded in the same space as the stored vectors and no endpoint URL
// ever appears in a query. `input` is
//  * a string literal -> embedded as TEXT, or
//  * an IRI           -> treated as an IMAGE URL (or `data:` URI), which the
//                        endpoint fetches and embeds.
// The result is the embedding as a comma-separated float-list string (a
// LocalVocab literal, shortest round-trippable floats) -- exactly the query
// vector format `vec:distance` parses, so the two compose:
//
//   BIND(vec:distance(<.../index/emb>, ?e,
//                     vec:embed(<.../index/emb>, "a red bicycle")) AS ?d)
//
// Results are MEMOIZED by input, so a constant input embeds exactly once per
// query (and a per-row variable embeds once per distinct value). NOTE: passing
// the query as raw bytes instead of a float-list string -- a query-side vector
// value that is serialized only on projection -- is a possible future
// optimization for a hot per-row embed; a constant query embeds once, so the
// string round trip is negligible there. Anything that is neither a literal
// nor an IRI (including UNDEF) evaluates to UNDEF. Throws if the index has no
// embedding endpoint configured.
class VectorEmbedExpression : public SparqlExpression {
 public:
  VectorEmbedExpression(std::string indexName, Ptr input);

  ExpressionResult evaluate(EvaluationContext* context) const override;
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override;

 private:
  ql::span<Ptr> childrenImpl() override;

  std::string indexName_;
  Ptr input_;
  // The memoized embeddings, keyed by kind-tagged input value. `mutable`
  // because `evaluate` is const but may be called once per block; an
  // expression instance is not evaluated concurrently (the same pattern as
  // `LiteralExpression`'s result cache).
  mutable ad_utility::HashMap<std::string, std::string> cache_;
};

// Factory used by the `SparqlFunctionRegistry` (arity 2: the index IRI
// `<.../vectorSearch/index/NAME>` and the input expression). Throws a
// user-facing error on a wrong arity or a malformed index argument.
SparqlExpression::Ptr makeVectorEmbedExpression(
    std::vector<SparqlExpression::Ptr> args);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTOREMBEDEXPRESSION_H
