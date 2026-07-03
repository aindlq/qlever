// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H

#include <optional>
#include <string>
#include <vector>

#include "rdfTypes/Variable.h"
#include "services/vectorSearch/VectorIndexFormat.h"

// Dependency-light configuration shared by the vector-search parser
// (`parsedQuery::VectorSearchQuery`) and the engine operation (`VectorSearch`).
// Kept free of engine includes so the parser can use it without a cycle
// (mirrors `SpatialJoinConfig.h`).
namespace qlever::vector {

// A standalone vector similarity search that produces a `(?result[, ?score])`
// top-k table. The query point is supplied either as an explicit vector or as a
// constant entity IRI whose vector is looked up in the index.
struct VectorSearchConfiguration {
  // Name of the vector index to search (must be loaded).
  std::string indexName_;

  // An image query point (embedded at query time). Only URLs (which the
  // embedding endpoint fetches) and base64 payloads are supported; reading
  // server-local files from query text would be an arbitrary-file-read
  // primitive for remote clients.
  enum class ImageKind { Url, Base64 };
  struct ImageQuery {
    ImageKind kind_;
    std::string value_;  // URL, or base64 (raw or data URI)
  };

  // The query point. Exactly one of these specifies it:
  std::optional<std::vector<float>> queryVector_;  // an explicit vector,
  std::optional<std::string> queryEntityIri_;      // a constant entity IRI,
  std::optional<std::string> queryText_;           // free text to embed,
  std::optional<ImageQuery> queryImage_;           // an image to embed, or
  std::optional<Variable> leftVariable_;           // a variable bound by the
                                                   // nested query pattern (the
  // "for each ?x" binary form).

  // The "among" form: an outer-bound candidate restriction. When set, the
  // candidate set (and the result) is this variable, bound by the SURROUNDING
  // query; the search returns the top-k of those candidates by distance to the
  // (fixed) query point. It always equals `resultVariable_` and is mutually
  // exclusive with `leftVariable_`. Handled by `VectorSearchAmong`.
  std::optional<Variable> amongVariable_;

  // The variable bound to each result entity.
  Variable resultVariable_{"?_qlever_internal_vec_result"};

  // Optional variable bound to the similarity distance of each result.
  std::optional<Variable> scoreVariable_;

  // Number of nearest neighbours to return.
  size_t k_ = 10;

  // Optional upper bound on the distance of returned neighbours.
  std::optional<float> maxDistance_;

  // Optional algorithm override: force exact or approximate search. If unset,
  // the index decides (HNSW if available, else exact).
  enum class Algorithm { Automatic, Exact, Hnsw };
  Algorithm algorithm_ = Algorithm::Automatic;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H
