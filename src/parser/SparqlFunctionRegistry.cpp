// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "parser/SparqlFunctionRegistry.h"

namespace parsedQuery {

// _____________________________________________________________________________
SparqlFunctionRegistry& SparqlFunctionRegistry::get() {
  static SparqlFunctionRegistry instance;
  return instance;
}

// _____________________________________________________________________________
void SparqlFunctionRegistry::addExact(std::string iri, Factory factory) {
  entries_.insert_or_assign(std::move(iri), std::move(factory));
}

// _____________________________________________________________________________
const SparqlFunctionRegistry::Factory* SparqlFunctionRegistry::lookup(
    std::string_view iri) const {
  auto it = entries_.find(iri);
  return it == entries_.end() ? nullptr : &it->second;
}

}  // namespace parsedQuery
