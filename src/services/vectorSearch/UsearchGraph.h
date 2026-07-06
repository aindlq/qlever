// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_USEARCHGRAPH_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_USEARCHGRAPH_H

#include <cstdint>
#include <usearch/index.hpp>
#include <usearch/index_plugins.hpp>

#include "services/vectorSearch/VectorIndexFormat.h"

// Internal shared pieces of the usearch integration (used by the builder and
// the reader; NOT part of the public interface of this folder).
//
// We use usearch's low-level `index_gt` (graph only) instead of the high-level
// `index_dense_gt`, because the latter insists on keeping its own copy of every
// vector in RAM and in its file. Our graph is keyed by the IMMUTABLE row index
// of the flat `.data` store, and distance computations read the vectors
// directly from that (memory-mapped) store: vectors exist exactly once -- on
// disk, in the page cache, and during the build. This is also what makes
// remapping after a knowledge-graph re-index cheap: the graph never references
// entity ids, only rows.
namespace qlever::vector {

namespace uu = unum::usearch;

// Keys are row indices of the flat store (NOT `ValueId` bits -- see above).
using GraphIndex = uu::index_gt<uu::distance_punned_t, std::uint64_t>;

// The metric passed to `index_gt`: resolves a graph member to its vector via
// the row-keyed flat store and delegates to the (SIMD-dispatched)
// `metric_punned_t`. Mirrors `index_dense_gt::metric_proxy_t`.
class FlatStoreMetric {
 public:
  using member_cref_t = GraphIndex::member_cref_t;
  using member_citerator_t = GraphIndex::member_citerator_t;

  // `data` is the base pointer of the row-major matrix (in the index's
  // storage scalar type) with `rowBytes` bytes per row and `numRows` rows;
  // `metric` computes the distance between two raw vector pointers of that
  // type. `numRows` is used to clamp graph member keys so that a corrupt
  // `.hnsw` (whose keys are supposed to be row indices in `[0, numRows)`)
  // cannot drive an out-of-bounds read of the flat store during traversal.
  FlatStoreMetric(const char* data, size_t rowBytes, size_t numRows,
                  const uu::metric_punned_t& metric) noexcept
      : data_{data}, rowBytes_{rowBytes}, numRows_{numRows}, metric_{&metric} {}

  uu::distance_punned_t operator()(const uu::byte_t* a,
                                   member_cref_t b) const noexcept {
    return f(a, v(b));
  }
  uu::distance_punned_t operator()(member_cref_t a,
                                   member_cref_t b) const noexcept {
    return f(v(a), v(b));
  }
  uu::distance_punned_t operator()(const uu::byte_t* a,
                                   member_citerator_t b) const noexcept {
    return f(a, v(*b));
  }
  uu::distance_punned_t operator()(member_citerator_t a,
                                   member_citerator_t b) const noexcept {
    return f(v(*a), v(*b));
  }
  uu::distance_punned_t operator()(const uu::byte_t* a,
                                   const uu::byte_t* b) const noexcept {
    return f(a, b);
  }

  const uu::byte_t* rowPtr(uint64_t row) const noexcept {
    return reinterpret_cast<const uu::byte_t*>(data_ + row * rowBytes_);
  }

 private:
  const uu::byte_t* v(member_cref_t m) const noexcept {
    uint64_t key = uu::get_key(m);
    // Clamp an out-of-range key (only possible from a corrupt `.hnsw`) to a
    // valid row so the read below stays in bounds; the resulting distance is
    // merely wrong for that corrupt node, never an OOB access.
    return rowPtr(key < numRows_ ? key : 0);
  }
  uu::distance_punned_t f(const uu::byte_t* a,
                          const uu::byte_t* b) const noexcept {
    return (*metric_)(a, b);
  }

  const char* data_;
  size_t rowBytes_;
  size_t numRows_;
  const uu::metric_punned_t* metric_;
};

// Map our enums to the corresponding usearch enum values.
inline uu::metric_kind_t toUsearchMetric(VectorMetric m) {
  switch (m) {
    case VectorMetric::Cosine:
      return uu::metric_kind_t::cos_k;
    case VectorMetric::L2Sq:
      return uu::metric_kind_t::l2sq_k;
    case VectorMetric::InnerProduct:
      return uu::metric_kind_t::ip_k;
  }
  return uu::metric_kind_t::cos_k;
}

inline uu::scalar_kind_t toUsearchScalar(VectorScalar s) {
  switch (s) {
    case VectorScalar::F32:
      return uu::scalar_kind_t::f32_k;
    case VectorScalar::F16:
      return uu::scalar_kind_t::f16_k;
    case VectorScalar::I8:
      return uu::scalar_kind_t::i8_k;
    case VectorScalar::Bf16:
      return uu::scalar_kind_t::bf16_k;
  }
  return uu::scalar_kind_t::f32_k;
}

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_USEARCHGRAPH_H
