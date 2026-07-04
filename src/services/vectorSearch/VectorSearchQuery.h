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

// The IRIs of the `vec:distance*` SPARQL function family: a per-row distance
// from a query point to `?entity`'s stored vector. BIND one and ORDER BY +
// LIMIT to run a filtered top-k search using QLever's own operators. The query
// point is a constant vector/entity (`vec:distance`), or free text / an image
// embedded at query time via the index's own endpoint (`vec:distanceText` /
// `vec:distanceImage`). Registered with the `SparqlFunctionRegistry`.
inline constexpr std::string_view VECTOR_DISTANCE_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/distance>";
inline constexpr std::string_view VECTOR_DISTANCE_TEXT_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/distanceText>";
inline constexpr std::string_view VECTOR_DISTANCE_IMAGE_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/distanceImage>";

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
