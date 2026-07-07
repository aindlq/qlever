// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H

#include <array>
#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "engine/sparqlExpressions/SparqlExpression.h"

namespace sparqlExpression {

// The `vec:distance` SPARQL expression: a per-row double distance between two
// vector SOURCES in the named vector index. Each source is an arbitrary
// sub-expression that evaluates (per row) to either
//  * an ENTITY (IRI/id) -> its stored vector is looked up in the index, or
//  * a comma-separated FLOAT-LIST literal -> parsed into a query vector:
//    either PLAIN (an inline "0.1,0.2,...", dimension-checked only) or TYPED
//    with an embedding space, `"..."^^<.../vec/MODEL/PRECISION>` (the output
//    of `vec:embed`/`vec:vector`), which is VALIDATED against the index --
//    the model (unless either side declares none), the precision, and the
//    dimension must all match, so a vector fetched from a DIFFERENT index via
//    `vec:vector` is safe: it computes iff the two indices share a space and
//    is `UNDEF` otherwise (see `VEC_QUERY_DATATYPE_PREFIX`).
// A row whose source is neither (including an entity without a live vector)
// evaluates to `UNDEF`. Combined with QLever's own `BIND`, `ORDER BY`, and
// `LIMIT`, this IS filtered top-k vector search -- no bespoke operator needed:
//
//   BIND(vec:distance(<.../index/clip>, ?e, "0.1,0.2,...") AS ?d)
//   FILTER(BOUND(?d)) ... ORDER BY ?d LIMIT k
//
// This makes entity<->entity (`vec:distance(<idx>, ?a, ?b)`), entity<->text
// (`vec:distance(<idx>, ?a, vec:embed(<idx>, "cat"))`), and inline query
// vectors all work through ONE composable function; the metric is a property
// of the index, so distances are uniformly smaller-is-closer.
//
// IMPORTANT: keep the `FILTER(BOUND(?d))` when a source may be UNDEF for some
// rows. `UNDEF` sorts BEFORE every real value in ascending `ORDER BY`, so
// without the filter a `LIMIT k` would return sourceless rows first instead of
// the k nearest. The filter is unnecessary only when every row is known to
// resolve.
//
// A source whose sub-expression is CONSTANT (a string literal, a constant
// IRI, a `vec:embed` of constants, ...) is resolved ONCE -- the expression
// framework yields a constant result for a constant sub-expression, and this
// expression branches on that -- so a constant query point is parsed/encoded a
// single time, not per row. The first argument is the index IRI
// `<.../vectorSearch/index/NAME>`: the SAME resource the auto-materialized
// metadata triples live on, so the thing you search is the thing you can
// introspect with plain SPARQL. This lives entirely in the vector-search
// service folder and is wired into the parser via the generic
// `parsedQuery::SparqlFunctionRegistry`.
class VectorDistanceExpression : public SparqlExpression {
 public:
  // `source1`/`source2` are the two vector-source operand expressions,
  // evaluated per row (each typically a variable, a constant, or `vec:embed`).
  VectorDistanceExpression(std::string indexName, Ptr source1, Ptr source2);

  // Logs ONE accumulated `vec:distance` timing line for the whole query
  // execution (see the accumulator members below); silent if this instance
  // never computed a distance (e.g. clones made during query planning, or
  // never-evaluated instances).
  ~VectorDistanceExpression() override;

  ExpressionResult evaluate(EvaluationContext* context) const override;
  std::string getCacheKey(const VariableToColumnMap& varColMap) const override;

 private:
  ql::span<Ptr> childrenImpl() override;

  std::string indexName_;
  std::array<Ptr, 2> sources_;

  // Accumulated distance-timing stats across ALL `evaluate()` blocks of one
  // query execution: Bind's lazy path calls `evaluate()` once per input chunk
  // (10000-row chunks) on the SAME expression object, so logging per block
  // would emit hundreds of near-identical lines. The destructor logs the total
  // ONCE when the query tree is torn down. Mutable because `evaluate()` is
  // const; PLAIN (not atomic) because one expression instance is never
  // evaluated by two threads at once -- the parallelism is WITHIN a block and
  // joins before the accumulation happens.
  mutable size_t totalDistances_ = 0;
  mutable std::chrono::microseconds totalDistanceTime_{0};
  mutable size_t distanceBlocks_ = 0;
  // The index shape, cached on the first block that computes anything, so the
  // destructor can log it without re-resolving the index.
  mutable size_t loggedDim_ = 0;
  mutable std::string loggedScalar_;
};

// Shared front of the vector-search function factories: extract the index
// name from `arg`, which must be a constant IRI of the form
// `<.../vectorSearch/index/NAME>` (see `VECTOR_METADATA_SUBJECT_PREFIX`).
// Throws a user-facing error (mentioning `functionName`) otherwise.
std::string parseVectorIndexIriArgument(const SparqlExpression* arg,
                                        std::string_view functionName);

// Factory used by the `SparqlFunctionRegistry` (arity 3: the index IRI and the
// two vector-source expressions). Throws a user-facing error on a wrong arity
// or a malformed index argument.
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args);

}  // namespace sparqlExpression

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORDISTANCEEXPRESSION_H
