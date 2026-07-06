// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>

#include "services/vectorSearch/VectorIndex.h"
#include "util/HashMap.h"

class Index;

namespace qlever::vector {

// The name under which the loaded vector indices are stored on the `IndexImpl`
// via the generic extension mechanism (see `index/IndexExtension.h`).
inline constexpr std::string_view VECTOR_EXTENSION_NAME = "vectorSearch";

// The auto-materialized metadata triples that make every loaded vector index
// queryable as an RDF resource (see
// `docs/vector-index/index-payload-design.md`, section "idx: metadata
// triples"): the load hook inserts, once per index,
//   <.../vectorSearch/index/NAME>  vec:dimension  3 ;
//       vec:metric "cosine" ; vec:precision "f32" ; vec:count 4 ;
//       vec:model "clip" .          # only if an embedding model is configured
// as DELTA triples. The IRIs live in the public `vectorSearch` namespace (NOT
// in QLever's internal `builtin-functions/` namespace), so the triples behave
// like ordinary inserted data and are visible to plain SPARQL.
inline constexpr std::string_view VECTOR_METADATA_SUBJECT_PREFIX =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/";
inline constexpr std::string_view VECTOR_METADATA_DIMENSION_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/dimension>";
inline constexpr std::string_view VECTOR_METADATA_METRIC_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/metric>";
inline constexpr std::string_view VECTOR_METADATA_PRECISION_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision>";
inline constexpr std::string_view VECTOR_METADATA_COUNT_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/count>";
inline constexpr std::string_view VECTOR_METADATA_MODEL_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/model>";

// All vector indices of a database, keyed by name. This is the object stored as
// the "vectorSearch" index extension and retrieved at query time.
class VectorIndexCollection {
 public:
  void add(const std::string& name, VectorIndex index) {
    indices_.insert_or_assign(name, std::move(index));
  }
  const VectorIndex* get(const std::string& name) const {
    auto it = indices_.find(name);
    return it == indices_.end() ? nullptr : &it->second;
  }

 private:
  ad_utility::HashMap<std::string, VectorIndex> indices_;
};

// Convenience for operations: the loaded `VectorIndex` named `name`, or an
// empty pointer. The returned `shared_ptr` aliases the whole collection, so the
// index stays valid even if the extension is ever replaced while in use.
std::shared_ptr<const VectorIndex> getVectorIndex(const Index& index,
                                                  const std::string& name);

// Recompute the entity mapping of the existing on-disk vector index `name`
// against the CURRENTLY LOADED knowledge graph: re-resolves the row-aligned
// `.iris` sidecar (in parallel; `numThreads` 0 = all cores) and rewrites
// `.keys`/`.rowmap`/`.meta`. The vector data and the HNSW graph are reused
// as-is, which makes this orders of magnitude cheaper than a rebuild after the
// RDF data was re-indexed. Entities that are no longer part of the knowledge
// graph become tombstones. Returns (live entities, tombstones). Also invoked
// via `--service-index '{"vectorSearch":[{"name":"...","remap":true}]}'`.
std::pair<uint64_t, uint64_t> remapVectorIndex(const Index& index,
                                               const std::string& basename,
                                               const std::string& name,
                                               uint32_t numThreads = 0);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
