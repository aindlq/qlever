// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#ifndef QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXBUILDER_H
#define QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXBUILDER_H

#include <cstdint>
#include <span>
#include <string>
#include <vector>

#include "global/Id.h"
#include "services/vectorSearch/VectorIndexFormat.h"

namespace qlever::vector {

// Builds the on-disk files of a single named vector index (see
// `VectorIndexFormat.h` for the layout). Usage:
//
//   VectorIndexBuilder b{basename, config};
//   for (each (entity, vector)) b.add(entity, vector);
//   VectorIndexMetadata meta = b.build();   // writes .keys/.data/.meta[/.hnsw]
//
// The builder accumulates vectors in memory, sorts them by entity id (so the
// reader can binary-search), writes the flat store, and -- if
// `config.buildHnsw` -- builds and saves a usearch HNSW index. (Streaming /
// external-memory building is a later optimisation; see the implementation
// log.)
class VectorIndexBuilder {
 public:
  VectorIndexBuilder(std::string basename, VectorIndexConfig config);

  // Add one entity's vector. `vector.size()` must equal `config.dimensions`.
  // Duplicate entities are not deduplicated here (the last one wins at lookup
  // is NOT guaranteed); callers should provide each entity at most once.
  void add(Id entity, std::span<const float> vector);

  // Number of vectors added so far.
  size_t size() const { return ids_.size(); }

  // Finalize and write all files. Returns the metadata that was persisted.
  VectorIndexMetadata build();

 private:
  std::string basename_;
  VectorIndexConfig config_;
  std::vector<uint64_t> ids_;   // entity ids, insertion order
  std::vector<float> data_;     // row-major, stride = config.dimensions
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_INDEX_VECTORINDEX_VECTORINDEXBUILDER_H
