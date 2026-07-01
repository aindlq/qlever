// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_ENGINE_MAGICSERVICEPLANNING_H
#define QLEVER_SRC_ENGINE_MAGICSERVICEPLANNING_H

#include <functional>
#include <memory>
#include <typeindex>

#include "util/HashMap.h"

class Operation;
class QueryExecutionContext;
class QueryExecutionTree;
namespace parsedQuery {
struct MagicServiceQuery;
class GraphPattern;
}  // namespace parsedQuery

// Stable, public interface that a magic-`SERVICE` planner handler uses to build
// its query plan, without depending on the planner's private internals
// (`GraphPatternPlanner`, `makeSubtreePlan`, `SubtreePlan`). A concrete
// implementation is provided inside `QueryPlanner.cpp`. This lets a custom
// service register its planning logic from its own translation unit (e.g. in
// `src/services/<name>/`) with no edits to the planner.
class MagicServicePlanningContext {
 public:
  virtual ~MagicServicePlanningContext() = default;

  // The execution context to construct operations with.
  virtual QueryExecutionContext* qec() const = 0;

  // Add a leaf operation (no children) as a candidate result for this service.
  virtual void addLeafOperation(std::shared_ptr<Operation> operation) = 0;

  // Plan the nested `childPattern`; for each resulting candidate child subtree,
  // call `makeOperation(child)` to build the service's operation, and add it as
  // a candidate. Use this for services whose operation wraps a nested pattern.
  // The pattern is copied before planning, so the parsed query stays unchanged
  // (it may be aliased by other copies of the `ParsedQuery`).
  virtual void addOperationWithChildPattern(
      const parsedQuery::GraphPattern& childPattern,
      std::function<
          std::shared_ptr<Operation>(std::shared_ptr<QueryExecutionTree>)>
          makeOperation) = 0;
};

// A handler that turns a parsed magic-service query into query plan candidates
// using the provided context.
using MagicServicePlanner = std::function<void(
    MagicServicePlanningContext&, parsedQuery::MagicServiceQuery&)>;

// Registry of planner handlers, keyed by the concrete `MagicServiceQuery`
// subtype. Services self-register (typically from a static initializer in their
// own translation unit); the planner looks the handler up generically. Mirrors
// the parser-side `MagicServiceRegistry`.
class MagicServicePlannerRegistry {
 public:
  static MagicServicePlannerRegistry& get();
  void add(std::type_index type, MagicServicePlanner planner);
  // The handler for `type`, or `nullptr` if none is registered.
  const MagicServicePlanner* lookup(std::type_index type) const;

 private:
  ad_utility::HashMap<std::type_index, MagicServicePlanner> planners_;
};

#endif  // QLEVER_SRC_ENGINE_MAGICSERVICEPLANNING_H
