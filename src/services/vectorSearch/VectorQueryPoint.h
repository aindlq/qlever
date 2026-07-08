// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H

#include <optional>
#include <string_view>
#include <variant>
#include <vector>

#include "global/Id.h"
#include "services/vectorSearch/VectorSearchConfig.h"
#include "util/CancellationHandle.h"

class IndexImpl;

namespace qlever::vector {

class VectorIndex;

// The resolved query point of a vector search:
//  * `monostate`      -> the constant query entity is unknown or has no vector
//                        (the caller should return an empty result);
//  * `vector<float>`  -> an explicit or embedded query vector
//  (dimension-checked
//                        against the index);
//  * `Id`             -> a constant query ENTITY that has a stored vector (to
//  be
//                        searched by its stored vector directly, no f32 round
//                        trip).
using QueryPoint = std::variant<std::monostate, std::vector<float>, Id>;

// Resolve the query point of `config` against the index `vidx` (embedding text
// or an image via the index's endpoint if needed). Throws on a dimension
// mismatch or a missing embedding endpoint. Used by the `VectorSearch`
// operation (the SERVICE's `vec:queryText`/`vec:imageUrl` surface); the
// function surface embeds via `vec:embed` instead.
QueryPoint resolveQueryPoint(const VectorSearchConfiguration& config,
                             const VectorIndex& vidx, const IndexImpl& index,
                             ad_utility::SharedCancellationHandle handle);

// Log one phase of a vector SERVICE search at INFO (mirroring the `vec:distance`
// timing line), so a SERVICE query reports where its time went: query embedding,
// the brute-force/coarse scan, and the rerank pass. `numVectors`, when given, is
// the number of vectors that phase touched.
void logVectorSearchPhase(std::string_view indexName, std::string_view phase,
                          double milliseconds,
                          std::optional<size_t> numVectors = std::nullopt);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H
