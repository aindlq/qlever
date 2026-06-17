// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorIndexBuilder.h"

#include <algorithm>
#include <fstream>
#include <numeric>

#include <usearch/index_dense.hpp>

#include "util/Exception.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
namespace uu = unum::usearch;

// Keyed by raw 64-bit `ValueId` bits -> must be uint64 (see VectorIndex.cpp).
using DenseIndex = uu::index_dense_gt<std::uint64_t>;

uu::metric_kind_t toUsearchMetric(VectorMetric m) {
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
}  // namespace

VectorIndexBuilder::VectorIndexBuilder(std::string basename,
                                       VectorIndexConfig config)
    : basename_{std::move(basename)}, config_{std::move(config)} {
  AD_CONTRACT_CHECK(config_.dimensions > 0,
                    "A vector index needs a positive dimension.");
  AD_CONTRACT_CHECK(config_.scalar == VectorScalar::F32,
                    "Only f32 vectors are supported so far.");
}

void VectorIndexBuilder::add(Id entity, std::span<const float> vector) {
  AD_CONTRACT_CHECK(vector.size() == config_.dimensions,
                    "Vector dimension does not match the index configuration.");
  ids_.push_back(entity.getBits());
  data_.insert(data_.end(), vector.begin(), vector.end());
}

VectorIndexMetadata VectorIndexBuilder::build() {
  const size_t n = ids_.size();
  const size_t dim = config_.dimensions;

  // 1. Sort by entity id so the reader can binary-search. We sort an index
  //    permutation and then materialise the reordered keys + data.
  std::vector<size_t> perm(n);
  std::iota(perm.begin(), perm.end(), size_t{0});
  std::sort(perm.begin(), perm.end(),
            [&](size_t a, size_t b) { return ids_[a] < ids_[b]; });

  // 2. Write the keys file.
  {
    ad_utility::MmapVector<uint64_t> keys;
    keys.open(n, vectorKeysFile(basename_, config_.name));
    for (size_t i = 0; i < n; ++i) {
      keys[i] = ids_[perm[i]];
    }
    // Destructor flushes and writes the trailer.
  }

  // 3. Write the data file (row-major, in the sorted order).
  {
    ad_utility::MmapVector<float> data;
    data.open(n * dim, vectorDataFile(basename_, config_.name));
    for (size_t i = 0; i < n; ++i) {
      const float* src = data_.data() + perm[i] * dim;
      std::copy(src, src + dim, &data[i * dim]);
    }
  }

  // 4. Optionally build and save the HNSW index, keyed by entity id.
  bool hasHnsw = false;
  if (config_.buildHnsw && n > 0) {
    uu::metric_punned_t metric{dim, toUsearchMetric(config_.metric),
                               uu::scalar_kind_t::f32_k};
    uu::index_dense_config_t cfg;
    cfg.connectivity = config_.hnswConnectivity;
    cfg.expansion_add = config_.hnswExpansionAdd;
    cfg.expansion_search = config_.hnswExpansionSearch;
    auto made = DenseIndex::make(metric, cfg);
    AD_CONTRACT_CHECK(static_cast<bool>(made),
                      "Could not create the HNSW index: ", made.error.what());
    DenseIndex index = std::move(made.index);
    index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const float* src = data_.data() + perm[i] * dim;
      auto added = index.add(static_cast<DenseIndex::vector_key_t>(
                                 ids_[perm[i]]),
                             src);
      AD_CONTRACT_CHECK(static_cast<bool>(added),
                        "Could not add a vector to the HNSW index: ",
                        added.error.what());
    }
    auto saved = index.save(vectorHnswFile(basename_, config_.name).c_str());
    AD_CONTRACT_CHECK(static_cast<bool>(saved),
                      "Could not save the HNSW index: ", saved.error.what());
    hasHnsw = true;
  }

  // 5. Write the metadata file.
  VectorIndexMetadata meta;
  meta.config = config_;
  meta.numVectors = n;
  meta.hasHnsw = hasHnsw;
  meta.version = VECTOR_INDEX_VERSION;
  std::ofstream metaOut{vectorMetaFile(basename_, config_.name)};
  AD_CONTRACT_CHECK(metaOut.is_open(), "Could not write vector metadata file ",
                    vectorMetaFile(basename_, config_.name));
  metaOut << nlohmann::json(meta).dump(2);
  return meta;
}

}  // namespace qlever::vector
