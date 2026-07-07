// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_ENGINE_MAGICPREDICATEREGISTRY_H
#define QLEVER_SRC_ENGINE_MAGICPREDICATEREGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <string_view>

#include "util/HashMap.h"

class Operation;
class QueryExecutionContext;
class TripleComponent;

// A handler that turns a single "magic predicate" triple -- one whose constant
// predicate IRI is registered here -- into ONE leaf `Operation`, given the
// triple's `subject` and `object` components. The handler validates the
// triple's shape (which position must be a constant IRI and which a variable,
// that a referenced index exists, ...) and throws a clear user-facing error
// otherwise. Mirrors the SERVICE-side `MagicServicePlanner`, but keyed by the
// predicate IRI of an ORDINARY triple in a basic graph pattern instead of a
// `SERVICE` clause.
using MagicPredicatePlanner = std::function<std::shared_ptr<Operation>(
    QueryExecutionContext*, const TripleComponent& subject,
    const TripleComponent& object)>;

// Registry of magic-predicate handlers, keyed by the exact predicate IRI (the
// angle-bracketed string `SparqlTriple::getSimplePredicate()` returns). A
// plugin self-registers a handler from its own translation unit (typically a
// static initializer, e.g. in `src/services/<name>/`); the planner
// (`QueryPlanner::seedWithScansAndText`) looks the handler up generically by
// the triple's predicate IRI and, if found, builds that leaf operation instead
// of an ordinary index scan. This keeps the planner free of any per-plugin
// code -- exactly like the parser-side `MagicServiceRegistry` and the
// engine-side `MagicServicePlannerRegistry` do for `SERVICE` clauses.
class MagicPredicateRegistry {
 public:
  static MagicPredicateRegistry& get();

  // Register `planner` for the exact predicate `iri` (angle-bracketed).
  void add(std::string iri, MagicPredicatePlanner planner);

  // The handler registered for the predicate `iri`, or `nullptr` if none.
  const MagicPredicatePlanner* lookup(const std::string& iri) const;

 private:
  ad_utility::HashMap<std::string, MagicPredicatePlanner> planners_;
};

#endif  // QLEVER_SRC_ENGINE_MAGICPREDICATEREGISTRY_H
