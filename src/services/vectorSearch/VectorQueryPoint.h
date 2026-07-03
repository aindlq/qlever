// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H

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
// mismatch or a missing embedding endpoint. Shared by `VectorSearch` (whole
// index / nested candidate set) and `VectorSearchAmong` (outer-bound candidate
// set) so there is a single implementation of the query-point handling.
QueryPoint resolveQueryPoint(const VectorSearchConfiguration& config,
                             const VectorIndex& vidx, const IndexImpl& index,
                             ad_utility::SharedCancellationHandle handle);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORQUERYPOINT_H
