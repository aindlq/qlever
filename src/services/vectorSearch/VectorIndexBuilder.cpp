// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndexBuilder.h"

#include <filesystem>
#include <fstream>
#include <numeric>
#include <usearch/index_dense.hpp>

#include "backports/algorithm.h"
#include "util/Exception.h"
#include "util/Log.h"
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

// Removes the given files on destruction unless `dismiss()` was called.
// Used so that a failed build does not leave temporary files behind.
class FileCleanup {
 public:
  void track(std::string path) { paths_.push_back(std::move(path)); }
  void dismiss() { paths_.clear(); }
  ~FileCleanup() {
    for (const auto& path : paths_) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }

 private:
  std::vector<std::string> paths_;
};
}  // namespace

// ____________________________________________________________________________
VectorIndexBuilder::VectorIndexBuilder(std::string basename,
                                       VectorIndexConfig config)
    : basename_{std::move(basename)}, config_{std::move(config)} {
  if (config_.dimensions_ == 0) {
    AD_THROW("A vector index needs a positive dimension.");
  }
  if (config_.scalar_ != VectorScalar::F32) {
    AD_THROW("Only f32 vectors are supported so far.");
  }
}

// ____________________________________________________________________________
void VectorIndexBuilder::add(Id entity, ql::span<const float> vector) {
  if (vector.size() != config_.dimensions_) {
    AD_THROW("A vector with dimension " + std::to_string(vector.size()) +
             " was added to vector index \"" + config_.name_ +
             "\", which is configured with dimension " +
             std::to_string(config_.dimensions_) + ".");
  }
  ids_.push_back(entity.getBits());
  data_.insert(data_.end(), vector.begin(), vector.end());
}

// ____________________________________________________________________________
VectorIndexMetadata VectorIndexBuilder::build() {
  const size_t dim = config_.dimensions_;

  // 1. Sort by entity id so the reader can binary-search. We sort an index
  //    permutation and then materialise the reordered keys + data. Ties (=
  //    duplicate entities) are broken by insertion order, which both makes the
  //    sort deterministic and lets the dedup below keep the FIRST vector.
  std::vector<size_t> perm(ids_.size());
  std::iota(perm.begin(), perm.end(), size_t{0});
  ql::ranges::sort(perm, [&](size_t a, size_t b) {
    return ids_[a] != ids_[b] ? ids_[a] < ids_[b] : a < b;
  });

  // 2. Deduplicate: the HNSW index rejects duplicate keys, and duplicate rows
  //    in the flat store would yield duplicate join results. Keep the first
  //    vector of each entity and warn (duplicate input rows are common in ML
  //    pipeline output and must not abort a multi-hour index build).
  std::vector<size_t> rows;
  rows.reserve(perm.size());
  for (size_t i = 0; i < perm.size(); ++i) {
    if (i > 0 && ids_[perm[i]] == ids_[perm[i - 1]]) {
      continue;
    }
    rows.push_back(perm[i]);
  }
  if (rows.size() != perm.size()) {
    AD_LOG_WARN << "Vector index \"" << config_.name_ << "\": the input "
                << "contained " << (perm.size() - rows.size())
                << " duplicate entities; keeping the first vector of each."
                << std::endl;
  }
  const size_t n = rows.size();

  // All files are first written with a `.tmp` suffix and renamed into place at
  // the very end -- metadata LAST, because the presence of a consistent `.meta`
  // is what makes the loader consider the index. An interrupted or failed
  // build therefore never clobbers a previously working index, and re-running
  // the build always repairs the state.
  const std::string keysPath = vectorKeysFile(basename_, config_.name_);
  const std::string dataPath = vectorDataFile(basename_, config_.name_);
  const std::string hnswPath = vectorHnswFile(basename_, config_.name_);
  const std::string metaPath = vectorMetaFile(basename_, config_.name_);
  auto tmp = [](const std::string& path) { return path + ".tmp"; };
  FileCleanup cleanup;

  // 3. Write the keys file.
  {
    cleanup.track(tmp(keysPath));
    ad_utility::MmapVector<uint64_t> keys;
    keys.open(n, tmp(keysPath));
    for (size_t i = 0; i < n; ++i) {
      keys[i] = ids_[rows[i]];
    }
    // Destructor flushes and writes the trailer.
  }

  // 4. Write the data file (row-major, in the sorted order).
  {
    cleanup.track(tmp(dataPath));
    ad_utility::MmapVector<float> data;
    data.open(n * dim, tmp(dataPath));
    for (size_t i = 0; i < n; ++i) {
      const float* src = data_.data() + rows[i] * dim;
      std::copy(src, src + dim, &data[i * dim]);
    }
  }

  // 5. Optionally build and save the HNSW index, keyed by entity id.
  bool hasHnsw = config_.buildHnsw_ && n > 0;
  if (hasHnsw) {
    cleanup.track(tmp(hnswPath));
    uu::metric_punned_t metric{dim, toUsearchMetric(config_.metric_),
                               uu::scalar_kind_t::f32_k};
    uu::index_dense_config_t cfg;
    cfg.connectivity = config_.hnswConnectivity_;
    cfg.expansion_add = config_.hnswExpansionAdd_;
    cfg.expansion_search = config_.hnswExpansionSearch_;
    auto made = DenseIndex::make(metric, cfg);
    if (!made) {
      AD_THROW("Could not create the HNSW index for vector index \"" +
               config_.name_ + "\": " + made.error.what());
    }
    DenseIndex index = std::move(made.index);
    index.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const float* src = data_.data() + rows[i] * dim;
      auto added =
          index.add(static_cast<DenseIndex::vector_key_t>(ids_[rows[i]]), src);
      if (!added) {
        AD_THROW("Could not add a vector to the HNSW index of vector index \"" +
                 config_.name_ + "\": " + added.error.what());
      }
    }
    auto saved = index.save(tmp(hnswPath).c_str());
    if (!saved) {
      AD_THROW("Could not save the HNSW index of vector index \"" +
               config_.name_ + "\": " + saved.error.what());
    }
  }

  // 6. Write the metadata file.
  VectorIndexMetadata meta;
  meta.config_ = config_;
  meta.numVectors_ = n;
  meta.hasHnsw_ = hasHnsw;
  meta.version_ = VECTOR_INDEX_VERSION;
  meta.vocabSize_ = vocabSize_;
  {
    cleanup.track(tmp(metaPath));
    std::ofstream metaOut{tmp(metaPath)};
    if (!metaOut.is_open()) {
      AD_THROW("Could not write the vector metadata file " + tmp(metaPath));
    }
    metaOut << nlohmann::json(meta).dump(2);
  }

  // 7. Rename everything into place (metadata last, see above). A stale
  //    `.hnsw` file from a previous build is removed if this build has none.
  std::filesystem::rename(tmp(keysPath), keysPath);
  std::filesystem::rename(tmp(dataPath), dataPath);
  if (hasHnsw) {
    std::filesystem::rename(tmp(hnswPath), hnswPath);
  } else {
    std::error_code ec;
    std::filesystem::remove(hnswPath, ec);
  }
  std::filesystem::rename(tmp(metaPath), metaPath);
  cleanup.dismiss();
  return meta;
}

}  // namespace qlever::vector
