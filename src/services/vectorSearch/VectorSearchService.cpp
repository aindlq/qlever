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

#include <memory>
#include <typeindex>

#include "engine/MagicServicePlanning.h"
#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "parser/MagicServiceQuery.h"
#include "parser/MagicServiceRegistry.h"
#include "parser/SparqlFunctionRegistry.h"
#include "services/vectorSearch/VectorDistanceExpression.h"
#include "services/vectorSearch/VectorSearch.h"
#include "services/vectorSearch/VectorSearchJoin.h"
#include "services/vectorSearch/VectorSearchQuery.h"

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
              "(`vec:left ?x`), or rank an existing candidate set with the "
              "`vec:distance` function + `ORDER BY` + `LIMIT`."};
        }
        if (config.leftVariable_.has_value()) {
          // "for each ?x bound by the SURROUNDING query, find the k nearest":
          // an incomplete join leaf; the planner's join enumeration adds the
          // subtree that binds `<left>` (see `IncompleteJoinOperation`).
          ctx.addLeafOperation(std::make_shared<VectorSearchJoin>(qec, config));
        } else {
          // Whole-index query point -> top-k table.
          ctx.addLeafOperation(std::make_shared<VectorSearch>(qec, config));
        }
      });

  // 3. Parser: register the `vec:distance*` SPARQL functions so that
  //    `BIND(vec:distance(...) AS ?d) ... ORDER BY ?d LIMIT k` ranks an
  //    existing set with QLever's own operators. `vec:distanceText`/`Image`
  //    embed the query point at query time via the index's own endpoint. The
  //    factories live in this folder; the parser only sees the generic
  //    registry.
  auto& functions = parsedQuery::SparqlFunctionRegistry::get();
  functions.addExact(std::string{parsedQuery::VECTOR_DISTANCE_IRI},
                     &sparqlExpression::makeVectorDistanceExpression);
  functions.addExact(std::string{parsedQuery::VECTOR_DISTANCE_TEXT_IRI},
                     &sparqlExpression::makeVectorDistanceTextExpression);
  functions.addExact(std::string{parsedQuery::VECTOR_DISTANCE_IMAGE_IRI},
                     &sparqlExpression::makeVectorDistanceImageExpression);
}

// Run the registration at static-initialization time. The folder is linked as
// an OBJECT library, so this initializer is always included (not stripped).
[[maybe_unused]] const bool vectorSearchRegistered = [] {
  registerVectorSearchService();
  return true;
}();

}  // namespace
