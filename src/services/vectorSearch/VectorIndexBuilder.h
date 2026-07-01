// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H

#include <cstdint>
#include <fstream>
#include <string>
#include <string_view>
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
//   for (each row) b.add(entity, iri, vector);
//   VectorIndexMetadata meta = b.build();   // writes all index files
//
// Designed for very large inputs: `add` STREAMS the vectors (and IRIs) to
// temporary files, so the builder's resident memory is ~24 bytes per row
// regardless of the vector dimension (100M rows = ~2.4 GB), never the vectors
// themselves. `build` sorts the rows by entity id, gathers the flat store with
// parallel reads, and -- if `config.buildHnsw_` -- constructs the usearch HNSW
// graph CONCURRENTLY (`config.buildThreads_`, 0 = all cores) with the vectors
// read from the memory-mapped store (usearch holds only the graph in RAM).
//
// Duplicate entities are deduplicated (the first vector of each entity wins,
// with a warning). All files are written to temporaries and renamed into place
// at the end (metadata last), so an interrupted build never leaves a
// readable-but-inconsistent index behind.
//
// NOTE: a concurrently built HNSW graph is not bit-for-bit deterministic
// (insertion interleaving affects the neighbourhood choices); recall is
// unaffected. Set `buildThreads_ = 1` for a deterministic build.
class VectorIndexBuilder {
 public:
  VectorIndexBuilder(std::string basename, VectorIndexConfig config);

  // Record the size of the knowledge-graph vocabulary this index is built
  // against. Stored in the metadata and checked at load time, so that stale
  // vector files are not silently applied to a rebuilt main index.
  void setVocabSize(uint64_t vocabSize) { vocabSize_ = vocabSize; }

  // Add one entity's vector. `vector.size()` must equal `config.dimensions_`.
  // The `iri` is persisted row-aligned so that the index can be cheaply
  // remapped after the knowledge graph is re-indexed.
  void add(Id entity, std::string_view iri, ql::span<const float> vector);

  // Number of vectors added so far (before deduplication).
  size_t size() const { return ids_.size(); }

  // Finalize and write all files. Returns the metadata that was persisted.
  VectorIndexMetadata build();

 private:
  std::string basename_;
  VectorIndexConfig config_;
  uint64_t vocabSize_ = 0;

  // Per-row bookkeeping (the only per-row state held in RAM).
  std::vector<uint64_t> ids_;         // entity ids, insertion order
  std::vector<uint64_t> iriOffsets_;  // byte offset of the IRI in the temp file
  std::vector<uint32_t> iriLengths_;  // byte length of the IRI (no newline)

  // Unsorted temporary spill files fed by `add`.
  std::string vecSpillPath_;
  std::string iriSpillPath_;
  std::ofstream vecSpill_;
  std::ofstream iriSpill_;
  uint64_t iriSpillOffset_ = 0;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXBUILDER_H
