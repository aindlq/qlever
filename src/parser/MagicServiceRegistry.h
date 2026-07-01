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

// A registry of magic `SERVICE` types, so a new service can be added without
// editing the parser's dispatch, the `GraphPatternOperation` variant, or the
// exhaustiveness checks: a service registers (a) an IRI matcher and (b) a
// factory that creates its `MagicServiceQuery`. The parser looks the IRI up
// here and parses into a generic `MagicService` node (see
// `GraphPatternOperation.h`).
//
// Services self-register via a static initializer (see e.g.
// a service`s own .cpp). The registry is a function-local static, so
// registration order across translation units is safe.
class MagicServiceRegistry {
 public:
  using Iri = ad_utility::triple_component::Iri;
  // Matches against the IRI's string representation (with angle brackets).
  using Matcher = std::function<bool(std::string_view)>;
  // Builds a fresh, empty config object for the matched service. Receives the
  // service IRI (some services parse parameters out of it).
  using Factory = std::function<std::shared_ptr<MagicServiceQuery>(const Iri&)>;

  static MagicServiceRegistry& get();

  // Register a service. `matcher` decides whether a given service IRI belongs
  // to it; `factory` creates its (empty) query object.
  void add(Matcher matcher, Factory factory);

  // Convenience: register a service matched by an exact IRI string.
  void addExact(std::string_view exactIri, Factory factory);

  // The factory for the first service whose matcher accepts `iri`, or nullopt.
  std::optional<Factory> lookup(const Iri& iri) const;

 private:
  std::vector<std::pair<Matcher, Factory>> entries_;
};

}  // namespace parsedQuery

#endif  // QLEVER_SRC_PARSER_MAGICSERVICEREGISTRY_H
