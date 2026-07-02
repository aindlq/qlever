// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "engine/BuiltinMagicServicePlanners.h"

#include <memory>
#include <mutex>
#include <typeindex>

#include "engine/ExternalValues.h"
#include "engine/MagicServicePlanning.h"
#include "engine/NamedResultCache.h"
#include "engine/PathSearch.h"
#include "engine/QueryExecutionContext.h"
#include "engine/TextIndexScanForEntity.h"
#include "engine/TextIndexScanForWord.h"
#include "parser/ExternalValuesQuery.h"
#include "parser/NamedCachedResult.h"
#include "parser/PathQuery.h"
#include "parser/TextSearchQuery.h"
#include "util/Exception.h"
#include "util/TypeTraits.h"

namespace qlever {

namespace {
// Register a handler for the concrete `MagicServiceQuery` subtype `Query`.
template <typename Query, typename Handler>
void registerHandler(Handler handler) {
  MagicServicePlannerRegistry::get().add(
      std::type_index(typeid(Query)),
      [handler = std::move(handler)](MagicServicePlanningContext& ctx,
                                     parsedQuery::MagicServiceQuery& query) {
        handler(ctx, dynamic_cast<Query&>(query));
      });
}
}  // namespace

// _____________________________________________________________________________
void registerBuiltinMagicServicePlanners() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    // Path search: plan the nested pattern and wrap each candidate child in a
    // `PathSearch` (mirrors the former `visitPathSearch`).
    registerHandler<parsedQuery::PathQuery>([](MagicServicePlanningContext& ctx,
                                               parsedQuery::PathQuery& query) {
      auto config = query.toPathSearchConfiguration(ctx.qec()->getIndex());
      AD_CORRECTNESS_CHECK(query.childGraphPattern_.has_value());
      auto* qec = ctx.qec();
      ctx.addOperationWithChildPattern(
          query.childGraphPattern_.value(),
          [config = std::move(config),
           qec](std::shared_ptr<QueryExecutionTree> child)
              -> std::shared_ptr<Operation> {
            return std::make_shared<PathSearch>(qec, std::move(child), config);
          });
    });

    // Text search: emit one leaf scan (word or entity) per config, joined
    // together within the group (mirrors the former `visitTextSearch`).
    registerHandler<
        parsedQuery::TextSearchQuery>([](MagicServicePlanningContext& ctx,
                                         parsedQuery::TextSearchQuery& query) {
      auto* qec = ctx.qec();
      std::vector<std::shared_ptr<Operation>> operations;
      for (auto config : query.toConfigs(qec)) {
        operations.push_back(std::visit(
            [qec](auto& arg) -> std::shared_ptr<Operation> {
              using T = std::decay_t<decltype(arg)>;
              using Op = std::conditional_t<
                  ad_utility::isSimilar<T, TextIndexScanForEntityConfiguration>,
                  TextIndexScanForEntity, TextIndexScanForWord>;
              return std::make_shared<Op>(qec, std::move(arg));
            },
            config));
      }
      ctx.addSeedOperations(std::move(operations));
    });

    // External values: a single leaf operation.
    registerHandler<parsedQuery::ExternalValuesQuery>(
        [](MagicServicePlanningContext& ctx,
           parsedQuery::ExternalValuesQuery& query) {
          ctx.addLeafOperation(
              std::make_shared<ExternalValues>(ctx.qec(), query));
        });

    // Named cached result: a single leaf operation from the named-result cache.
    registerHandler<parsedQuery::NamedCachedResult>(
        [](MagicServicePlanningContext& ctx,
           parsedQuery::NamedCachedResult& query) {
          auto* qec = ctx.qec();
          ctx.addLeafOperation(
              qec->namedResultCache().getOperation(query.identifier(), qec));
        });
  });
}

}  // namespace qlever
