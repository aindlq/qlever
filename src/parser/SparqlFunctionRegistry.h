// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_PARSER_SPARQLFUNCTIONREGISTRY_H
#define QLEVER_SRC_PARSER_SPARQLFUNCTIONREGISTRY_H

#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "util/HashMap.h"

namespace sparqlExpression {
class SparqlExpression;
}

namespace parsedQuery {

// A registry of SPARQL function-call IRIs (e.g. the vector-search
// `vec:distance(...)` function), so a drop-in service under `src/services/` can
// add a custom function without editing the parser's hard-coded function
// dispatch (`SparqlQleverVisitor::processIriFunctionCall`). A service registers
// an exact function IRI plus a factory that turns the parsed argument
// expressions into a `SparqlExpression`. The parser consults this registry as a
// fallback, after its built-in `geof:`/`math:`/`xsd:`/`ql:` tables and before
// it reports an unknown function.
//
// This mirrors `MagicServiceRegistry` (which does the same for magic
// `SERVICE`s): the registry is a function-local static, so registration order
// across translation units is safe, and services self-register from a static
// initializer in their own object library.
class SparqlFunctionRegistry {
 public:
  using ExpressionPtr = std::unique_ptr<sparqlExpression::SparqlExpression>;
  using ArgList = std::vector<ExpressionPtr>;
  // Build a `SparqlExpression` from the parsed argument expressions. The
  // factory is responsible for checking the arity and the argument kinds; it
  // may throw (via `AD_THROW`) with a user-facing message on a misuse.
  using Factory = std::function<ExpressionPtr(ArgList)>;

  static SparqlFunctionRegistry& get();

  // Register `factory` for the exact function IRI `iri` (its bracketed string
  // representation, e.g. `<https://.../vectorSearch/distance>`). A second
  // registration for the same IRI overwrites the first.
  void addExact(std::string iri, Factory factory);

  // The factory registered for `iri` (bracketed representation), or `nullptr`.
  const Factory* lookup(std::string_view iri) const;

 private:
  ad_utility::HashMap<std::string, Factory> entries_;
};

}  // namespace parsedQuery

#endif  // QLEVER_SRC_PARSER_SPARQLFUNCTIONREGISTRY_H
