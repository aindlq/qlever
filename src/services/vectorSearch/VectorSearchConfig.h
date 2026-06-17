// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#ifndef QLEVER_SRC_ENGINE_VECTORSEARCHCONFIG_H
#define QLEVER_SRC_ENGINE_VECTORSEARCHCONFIG_H

#include <optional>
#include <string>
#include <vector>

#include "services/vectorSearch/VectorIndexFormat.h"
#include "rdfTypes/Variable.h"

// Dependency-light configuration shared by the vector-search parser
// (`parsedQuery::VectorSearchQuery`) and the engine operation (`VectorSearch`).
// Kept free of engine includes so the parser can use it without a cycle (mirrors
// `SpatialJoinConfig.h`).
namespace qlever::vector {

// A standalone vector similarity search that produces a `(?result[, ?score])`
// top-k table. The query point is supplied either as an explicit vector or as a
// constant entity IRI whose vector is looked up in the index. (The "join with a
// query variable bound outside the SERVICE" form is a separate, later
// operation.)
struct VectorSearchConfiguration {
  // Name of the vector index to search (must be loaded).
  std::string indexName_;

  // An image query point (embedded at query time).
  enum class ImageKind { Url, File, Base64 };
  struct ImageQuery {
    ImageKind kind_;
    std::string value_;  // URL, file path, or base64 (raw or data URI)
  };

  // The query point. Exactly one of these specifies it:
  std::optional<std::vector<float>> queryVector_;   // an explicit vector,
  std::optional<std::string> queryEntityIri_;       // a constant entity IRI,
  std::optional<std::string> queryText_;            // free text to embed,
  std::optional<ImageQuery> queryImage_;            // an image to embed, or
  std::optional<Variable> leftVariable_;            // a variable bound by the
                                                    // nested query pattern (the
                                                    // "for each ?x" binary form).

  // The variable bound to each result entity.
  Variable resultVariable_{"?_qlever_internal_vec_result"};

  // Optional variable bound to the similarity distance of each result.
  std::optional<Variable> scoreVariable_;

  // Number of nearest neighbours to return.
  size_t k_ = 10;

  // Optional algorithm override: force exact or approximate search. If unset,
  // the index decides (HNSW if available, else exact).
  enum class Algorithm { Automatic, Exact, Hnsw };
  Algorithm algorithm_ = Algorithm::Automatic;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_ENGINE_VECTORSEARCHCONFIG_H
