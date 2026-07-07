// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "engine/MagicPredicateRegistry.h"

// _____________________________________________________________________________
MagicPredicateRegistry& MagicPredicateRegistry::get() {
  static MagicPredicateRegistry instance;
  return instance;
}

// _____________________________________________________________________________
void MagicPredicateRegistry::add(std::string iri,
                                 MagicPredicatePlanner planner) {
  planners_.insert_or_assign(std::move(iri), std::move(planner));
}

// _____________________________________________________________________________
const MagicPredicatePlanner* MagicPredicateRegistry::lookup(
    const std::string& iri) const {
  auto it = planners_.find(iri);
  return it == planners_.end() ? nullptr : &it->second;
}
