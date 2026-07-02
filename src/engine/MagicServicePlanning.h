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
class Variable;
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

  // Add each operation as an INDEPENDENT seed of the current group's join
  // planning (they are joined together with the rest of the group, rather than
  // being alternatives for a single node). Use this for services that emit
  // several leaf scans to be combined (e.g. a text search's word/entity
  // scans).
  virtual void addSeedOperations(
      std::vector<std::shared_ptr<Operation>> operations) = 0;

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

// Interface for operations of registry-based magic services that take one side
// of a binary join from the SURROUNDING query (mirroring how `SpatialJoin` is
// planned): the handler adds the operation as an "incomplete" leaf whose
// variable-to-column map exposes the join variable as a possibly-undefined
// column; when the join enumeration meets a subtree that binds this variable,
// it completes the operation via `addJoinChild` instead of creating a normal
// join. An operation that is still incomplete at execution time must throw a
// clear user-facing error (the query then contains no other occurrence of the
// join variable).
class IncompleteJoinOperation {
 public:
  virtual ~IncompleteJoinOperation() = default;

  // False while the join child is still missing.
  virtual bool isJoinConstructed() const = 0;

  // The variable that the surrounding query has to bind.
  virtual const Variable& joinVariable() const = 0;

  // A NEW, completed operation with `child` -- a subtree that binds
  // `joinVariable()` -- as the missing side.
  virtual std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child) const = 0;
};

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
