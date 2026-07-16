// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_PARSER_MAGICSERVICEREGISTRY_H
#define QLEVER_SRC_PARSER_MAGICSERVICEREGISTRY_H

#include <functional>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

#include "parser/MagicServiceQuery.h"
#include "rdfTypes/Iri.h"

namespace parsedQuery {

// Registry of custom magic-`SERVICE` types, so an extension can add a new one
// without touching the parser's `SERVICE` dispatch or the
// `GraphPatternOperation` variant: it registers an IRI matcher and a factory
// that creates its `MagicServiceQuery`. The parser looks the IRI up here (after
// the built-in services) and parses into a generic `MagicService` node (see
// `GraphPatternOperation.h`). Consulted alongside `MagicServicePlannerRegistry`
// (engine), which turns the parsed query into an `Operation`.
class MagicServiceRegistry {
 public:
  using Iri = ad_utility::triple_component::Iri;
  // Matches against the IRI's bracketed string representation.
  using Matcher = std::function<bool(std::string_view)>;
  // Builds a fresh, empty config object for the matched service, given the
  // service IRI (some services parse parameters out of it).
  using Factory = std::function<std::shared_ptr<MagicServiceQuery>(const Iri&)>;

  static MagicServiceRegistry& get();

  void add(Matcher matcher, Factory factory);
  // Convenience: register a service matched by an exact IRI string.
  void addExact(std::string_view exactIri, Factory factory);

  // The factory of the first service whose matcher accepts `iri`, or nullopt.
  std::optional<Factory> lookup(const Iri& iri) const;

 private:
  std::vector<std::pair<Matcher, Factory>> entries_;
};

}  // namespace parsedQuery

#endif  // QLEVER_SRC_PARSER_MAGICSERVICEREGISTRY_H
