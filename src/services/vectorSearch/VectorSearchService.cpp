// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Self-registration of the vector-search magic SERVICE. This is the ONLY place
// that wires the service into QLever, and it touches no core engine/parser
// code:
//   * the parser factory is registered with `MagicServiceRegistry` (parser);
//   * the planner handler is registered with `MagicServicePlannerRegistry`
//     (engine) and uses only the stable `MagicServicePlanningContext` façade.
// The whole `src/services/vectorSearch/` folder can be added/removed without
// changing the variant, the parser/planner dispatch, or any CMake of the core
// libraries (see `src/services/CMakeLists.txt`).

#include <absl/strings/str_cat.h>

#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <typeindex>

#include "engine/MagicPredicateRegistry.h"
#include "engine/MagicServicePlanning.h"
#include "engine/Operation.h"
#include "engine/QueryExecutionContext.h"
#include "engine/QueryExecutionTree.h"
#include "index/Index.h"
#include "parser/MagicServiceQuery.h"
#include "parser/MagicServiceRegistry.h"
#include "parser/SparqlFunctionRegistry.h"
#include "parser/TripleComponent.h"
#include "services/vectorSearch/VectorDistanceExpression.h"
#include "services/vectorSearch/VectorEmbedExpression.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorMemberScan.h"
#include "services/vectorSearch/VectorSearch.h"
#include "services/vectorSearch/VectorSearchJoin.h"
#include "services/vectorSearch/VectorSearchQuery.h"
#include "services/vectorSearch/VectorVectorExpression.h"

namespace {

void registerVectorSearchService() {
  // 1. Parser: route `SERVICE <…/vectorSearch/>` to a `VectorSearchQuery`.
  parsedQuery::MagicServiceRegistry::get().addExact(
      parsedQuery::VECTOR_SEARCH_IRI,
      [](const ad_utility::triple_component::Iri&)
          -> std::shared_ptr<parsedQuery::MagicServiceQuery> {
        return std::make_shared<parsedQuery::VectorSearchQuery>();
      });

  // 2. Planner: turn a parsed `VectorSearchQuery` into a query plan, via the
  //    stable planning façade (no access to planner internals needed).
  MagicServicePlannerRegistry::get().add(
      std::type_index(typeid(parsedQuery::VectorSearchQuery)),
      [](MagicServicePlanningContext& ctx,
         parsedQuery::MagicServiceQuery& query) {
        auto& vectorQuery =
            dynamic_cast<parsedQuery::VectorSearchQuery&>(query);
        auto config = vectorQuery.toVectorSearchConfiguration();
        QueryExecutionContext* qec = ctx.qec();

        // The vector-search SERVICE is a pure "search" surface: config triples
        // on the blank node, no nested `{ ... }` pattern. Ranking an existing
        // candidate set is the `vec:distance` function (see below).
        if (vectorQuery.childGraphPattern_.has_value()) {
          throw std::runtime_error{
              "The vector-search SERVICE does not take a nested `{ ... }` "
              "pattern. Bind the query entity in the surrounding query "
              "(`vec:candidates ?x`, alias `vec:left`), or rank an existing "
              "candidate set with the `vec:distance` function + `ORDER BY` + "
              "`LIMIT`."};
        }
        if (config.leftVariable_.has_value()) {
          // "for each ?x bound by the SURROUNDING query, find the k nearest":
          // an incomplete join leaf; the planner's join enumeration adds the
          // subtree that binds `<left>` (see `IncompleteJoinOperation`).
          ctx.addLeafOperation(std::make_shared<VectorSearchJoin>(qec, config));
        } else {
          // Whole-index query point -> top-k table. This also covers
          // `<candidates>` TOGETHER with a query point: that form is lowered
          // to a produce config whose result variable IS the candidates
          // variable (see `toVectorSearchConfiguration`), so an unbound
          // candidates variable receives the whole-index top-k, and a bound
          // one is joined by the planner like any produce output.
          ctx.addLeafOperation(std::make_shared<VectorSearch>(qec, config));
        }
      });

  // 3. Parser: register the vector-search SPARQL functions so that
  //    `BIND(vec:distance(...) AS ?d) ... ORDER BY ?d LIMIT k` ranks an
  //    existing set with QLever's own operators. `vec:distance` takes two
  //    generalized vector sources (entities or float-list literals);
  //    `vec:embed` turns a text literal / image IRI into a TYPED float-list
  //    literal via the index's own endpoint, and `vec:vector` fetches an
  //    entity's STORED vector as the same typed literal (the validated
  //    cross-index bridge), so all three compose. The factories live in this
  //    folder; the parser only sees the generic registry.
  auto& functions = parsedQuery::SparqlFunctionRegistry::get();
  functions.addExact(std::string{parsedQuery::VECTOR_DISTANCE_IRI},
                     &sparqlExpression::makeVectorDistanceExpression);
  functions.addExact(std::string{parsedQuery::VECTOR_EMBED_IRI},
                     &sparqlExpression::makeVectorEmbedExpression);
  functions.addExact(std::string{parsedQuery::VECTOR_VECTOR_IRI},
                     &sparqlExpression::makeVectorVectorExpression);

  // 4. Planner: the `vec:hasMember` MAGIC PREDICATE. A triple
  //    `<.../index/NAME> vec:hasMember ?e` produces a `VectorMemberScan` leaf
  //    that emits the index's member entities as one already-`ValueId`-sorted
  //    column, so the planner merge-joins it with the surrounding query (no
  //    vectors are materialised). This drops the `FILTER(BOUND)` membership
  //    idiom: `?e a :T . vidx:X vec:hasMember ?e` is a clean, cheap join. The
  //    handler validates the triple's shape (constant index-IRI subject,
  //    variable object, index must exist) with clear user-facing errors.
  MagicPredicateRegistry::get().add(
      std::string{parsedQuery::VECTOR_HAS_MEMBER_IRI},
      [](QueryExecutionContext* qec, const TripleComponent& subject,
         const TripleComponent& object) -> std::shared_ptr<Operation> {
        if (!subject.isIri()) {
          throw std::runtime_error{
              "The subject of `vec:hasMember` must be a constant vector-index "
              "IRI `<https://qlever.cs.uni-freiburg.de/vectorSearch/index/"
              "NAME>`, not a variable or literal."};
        }
        std::optional<std::string> indexName =
            qlever::vector::indexNameFromMetadataIri(
                subject.getIri().toStringRepresentation());
        if (!indexName.has_value()) {
          throw std::runtime_error{absl::StrCat(
              "The subject of `vec:hasMember` must be a vector-index IRI "
              "`<https://qlever.cs.uni-freiburg.de/vectorSearch/index/NAME>`, "
              "but got ",
              subject.getIri().toStringRepresentation(), ".")};
        }
        if (!object.isVariable()) {
          throw std::runtime_error{
              "The object of `vec:hasMember` must be a variable that is bound "
              "to the members of the vector index."};
        }
        // The index must be loaded (it is at query time; this makes a typo in
        // the index IRI fail with a clear message rather than an empty result).
        if (!qlever::vector::getVectorIndex(qec->getIndex(),
                                            indexName.value())) {
          throw std::runtime_error{absl::StrCat(
              "There is no loaded vector index named '", indexName.value(),
              "'. Was the index built with `--service-index`?")};
        }
        return std::make_shared<VectorMemberScan>(
            qec, std::move(indexName).value(), object.getVariable());
      });
}

// Run the registration at static-initialization time. The folder is linked as
// an OBJECT library, so this initializer is always included (not stripped).
[[maybe_unused]] const bool vectorSearchRegistered = [] {
  registerVectorSearchService();
  return true;
}();

}  // namespace
