// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H

#include <cstdint>
#include <string>
#include <vector>

#include "backports/span.h"
#include "global/Id.h"
#include "services/vectorSearch/VectorIndexFormat.h"

namespace qlever::vector {

// Builds the on-disk files of a single named vector index (see
// `VectorIndexFormat.h` for the layout). Usage:
//
//   VectorIndexBuilder b{basename, config};
//   b.setVocabSize(vocabSize);              // fingerprint of the KG build
//   for (each (entity, vector)) b.add(entity, vector);
//   VectorIndexMetadata meta = b.build();   // writes .keys/.data/.meta[/.hnsw]
//
// The builder accumulates vectors in memory, sorts them by entity id (so the
// reader can binary-search), writes the flat store, and -- if
// `config.buildHnsw_` -- builds and saves a usearch HNSW index. Duplicate
// entities are deduplicated (the first vector of each entity wins, with a
// warning). All files are written to temporaries and renamed into place at the
// end (metadata last), so an interrupted build never leaves a
// readable-but-inconsistent index behind. (Streaming / external-memory building
// and a parallel HNSW build are follow-up optimisations; see the README.)
class VectorIndexBuilder {
 public:
  VectorIndexBuilder(std::string basename, VectorIndexConfig config);

  // Record the size of the knowledge-graph vocabulary this index is built
  // against. Stored in the metadata and checked at load time, so that stale
  // vector files are not silently applied to a rebuilt main index.
  void setVocabSize(uint64_t vocabSize) { vocabSize_ = vocabSize; }

  // Add one entity's vector. `vector.size()` must equal `config.dimensions_`.
  void add(Id entity, ql::span<const float> vector);

  // Number of vectors added so far (before deduplication).
  size_t size() const { return ids_.size(); }

  // Finalize and write all files. Returns the metadata that was persisted.
  VectorIndexMetadata build();

 private:
  std::string basename_;
  VectorIndexConfig config_;
  uint64_t vocabSize_ = 0;
  std::vector<uint64_t> ids_;  // entity ids, insertion order
  std::vector<float> data_;    // row-major, stride = config.dimensions_
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H
