// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHQUERY_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHQUERY_H

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "parser/MagicServiceQuery.h"
#include "services/vectorSearch/VectorSearchConfig.h"

namespace parsedQuery {

// The magic-service IRI that activates vector search. Defined in the service so
// the core parser has no knowledge of it.
inline constexpr std::string_view VECTOR_SEARCH_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/>";

// The IRIs of the vector-search SPARQL functions, registered with the
// `SparqlFunctionRegistry`:
//  * `vec:distance(<.../index/NAME>, s1, s2)` -- the per-row distance between
//    two vector sources (each an entity with a stored vector, or a
//    comma-separated float-list string, e.g. from `vec:embed`). BIND it and
//    ORDER BY + LIMIT to run a filtered top-k search using QLever's own
//    operators.
//  * `vec:embed(<.../index/NAME>, input)` -- embed a text literal or an image
//    IRI via the index's own configured endpoint and return the vector as a
//    typed float-list literal (composable with `vec:distance`).
//  * `vec:vector(<.../index/NAME>, entity)` -- the entity's STORED vector as
//    the same typed float-list literal, the bridge for validated CROSS-INDEX
//    distances (see `VectorVectorExpression`).
inline constexpr std::string_view VECTOR_DISTANCE_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/distance>";
inline constexpr std::string_view VECTOR_EMBED_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/embed>";
inline constexpr std::string_view VECTOR_VECTOR_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/vector>";

class VectorSearchException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

// Parses a `SERVICE <.../vectorSearch/>` clause into a
// `VectorSearchConfiguration`. Incrementally filled while parsing config
// triples (see `MagicServiceQuery`).
struct VectorSearchQuery : MagicServiceQuery {
  std::optional<std::string> indexName_;
  std::optional<std::vector<float>> queryVector_;
  std::optional<std::string> queryEntityIri_;
  std::optional<std::string> queryText_;  // free text -> embedded at query time
  std::optional<qlever::vector::VectorSearchConfiguration::ImageQuery>
      queryImage_;                   // image -> embedded at query time
  std::optional<Variable> leftVar_;  // binary "for each ?x" form: query entity
                                     // bound by the surrounding query
  std::optional<Variable> resultVar_;
  std::optional<Variable> scoreVar_;
  std::optional<size_t> k_;
  std::optional<float> maxDistance_;
  qlever::vector::VectorSearchConfiguration::Algorithm algo_ =
      qlever::vector::VectorSearchConfiguration::Algorithm::Automatic;

  // Inherited from `MagicServiceQuery`.
  void addParameter(const SparqlTriple& triple) override;
  void validate() const override;
  std::string_view name() const override { return "VectorSearch"; }

  // Lower to the engine configuration (also performs validation).
  qlever::vector::VectorSearchConfiguration toVectorSearchConfiguration() const;
};

}  // namespace parsedQuery

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHQUERY_H
