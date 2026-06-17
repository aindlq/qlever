// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "parser/MagicServiceRegistry.h"

namespace parsedQuery {

MagicServiceRegistry& MagicServiceRegistry::get() {
  static MagicServiceRegistry instance;
  return instance;
}

void MagicServiceRegistry::add(Matcher matcher, Factory factory) {
  entries_.emplace_back(std::move(matcher), std::move(factory));
}

void MagicServiceRegistry::addExact(std::string_view exactIri,
                                    Factory factory) {
  add([s = std::string{exactIri}](std::string_view iri) { return iri == s; },
      std::move(factory));
}

std::optional<MagicServiceRegistry::Factory> MagicServiceRegistry::lookup(
    const Iri& iri) const {
  std::string_view repr = iri.toStringRepresentation();
  for (const auto& [matcher, factory] : entries_) {
    if (matcher(repr)) {
      return factory;
    }
  }
  return std::nullopt;
}

}  // namespace parsedQuery
