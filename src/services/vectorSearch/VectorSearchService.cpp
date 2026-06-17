// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

// Self-registration of the vector-search magic SERVICE. This is the ONLY place
// that wires the service into QLever, and it touches no core engine/parser code:
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

        if (config.leftVariable_.has_value()) {
          // "for each ?x in the nested pattern, find the k nearest."
          ctx.addOperationWithChildPattern(
              vectorQuery.childGraphPattern_.value(),
              [config, qec](std::shared_ptr<QueryExecutionTree> child)
                  -> std::shared_ptr<Operation> {
                return std::make_shared<VectorSearchJoin>(qec, config,
                                                          std::move(child));
              });
        } else if (vectorQuery.childGraphPattern_.has_value()) {
          // Query point restricted to the candidate entities the nested pattern
          // binds (exact search over that set).
          ctx.addOperationWithChildPattern(
              vectorQuery.childGraphPattern_.value(),
              [config, qec](std::shared_ptr<QueryExecutionTree> child)
                  -> std::shared_ptr<Operation> {
                return std::make_shared<VectorSearch>(qec, config,
                                                      std::move(child));
              });
        } else {
          // Whole-index query point -> top-k table.
          ctx.addLeafOperation(std::make_shared<VectorSearch>(qec, config));
        }
      });
}

// Run the registration at static-initialization time. The folder is linked as an
// OBJECT library, so this initializer is always included (not stripped).
[[maybe_unused]] const bool vectorSearchRegistered = [] {
  registerVectorSearchService();
  return true;
}();

}  // namespace
