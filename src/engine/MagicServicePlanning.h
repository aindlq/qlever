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

// Interface for a binary-join operation that takes one (or both) of its join
// sides from the SURROUNDING query, planned by an "incomplete leaf" + join
// completion instead of a normal join. Both the built-in `SpatialJoin` and
// registry-based magic services (e.g. an outer-bound `VectorSearchJoin`)
// implement it, so `QueryPlanner` has a single generic completion path.
//
// The operation is planned as an incomplete leaf whose variable-to-column map
// exposes its still-missing join variable(s) as possibly-undefined columns;
// when the join enumeration meets a subtree that binds one of them, the planner
// completes the operation via `addJoinChild` instead of creating a normal join.
// An operation still incomplete at execution time must throw a clear
// user-facing error.
class IncompleteJoinOperation {
 public:
  virtual ~IncompleteJoinOperation() = default;

  // False while a required join child is still missing.
  virtual bool isJoinConstructed() const = 0;

  // True iff `var` is a still-missing join variable that the surrounding query
  // can bind to (partially) complete this operation. (An operation may expose
  // OUTPUT variables too; those are not join variables and return false, which
  // makes the planner complete via a join variable in another order.)
  virtual bool canBindJoinVariable(const Variable& var) const = 0;

  // When the operation is connected to the rest of the query through MORE than
  // one variable at once: if this returns true a normal join between the two
  // sides is acceptable (the planner falls through to it), otherwise the
  // connection is unsupported and the planner reports
  // `multipleJoinVariablesError()`. (True for a spatial join that merely
  // substitutes a geo filter; false for a service that requires completion.)
  virtual bool allowsNormalJoinOnMultipleVariables() const { return false; }

  // The user-facing error for an unsupported multi-variable connection.
  virtual std::string multipleJoinVariablesError() const = 0;

  // A NEW operation with `child` (which binds `var`, one of the missing join
  // variables) added as that side.
  virtual std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child, const Variable& var) const = 0;
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
