// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndexBuilder.h"

#include <atomic>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <thread>

#include "backports/algorithm.h"
#include "services/vectorSearch/UsearchGraph.h"
#include "util/Exception.h"
#include "util/File.h"
#include "util/Log.h"
#include "util/MmapVector.h"

namespace qlever::vector {

namespace {

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

// Run `job(threadIdx, firstRow, lastRow)` for a partition of `[0, n)` over
// `numThreads` threads and rethrow the first error, if any.
template <typename Job>
void parallelOverRows(size_t n, size_t numThreads, const Job& job) {
  numThreads = std::max<size_t>(1, std::min(numThreads, n));
  std::atomic_flag failed = ATOMIC_FLAG_INIT;
  std::mutex errorMutex;
  std::string firstError;
  auto guardedJob = [&](size_t t, size_t first, size_t last) {
    try {
      job(t, first, last);
    } catch (const std::exception& e) {
      if (!failed.test_and_set()) {
        std::lock_guard lock{errorMutex};
        firstError = e.what();
      }
    }
  };
  if (numThreads == 1) {
    guardedJob(0, 0, n);
  } else {
    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    size_t chunk = (n + numThreads - 1) / numThreads;
    for (size_t t = 0; t < numThreads; ++t) {
      size_t first = t * chunk;
      size_t last = std::min(n, first + chunk);
      if (first >= last) break;
      threads.emplace_back(guardedJob, t, first, last);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  {
    std::lock_guard lock{errorMutex};
    if (!firstError.empty()) {
      throw std::runtime_error{firstError};
    }
  }
}

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
  vecSpillPath_ = vectorDataFile(basename_, config_.name_) + ".spill";
  iriSpillPath_ = vectorIrisFile(basename_, config_.name_) + ".spill";
  vecSpill_.open(vecSpillPath_, std::ios::binary | std::ios::trunc);
  iriSpill_.open(iriSpillPath_, std::ios::binary | std::ios::trunc);
  if (!vecSpill_.is_open() || !iriSpill_.is_open()) {
    AD_THROW("Could not create temporary build files for vector index \"" +
             config_.name_ + "\" next to " + basename_);
  }
}

// ____________________________________________________________________________
void VectorIndexBuilder::add(Id entity, std::string_view iri,
                             ql::span<const float> vector) {
  if (vector.size() != config_.dimensions_) {
    AD_THROW("A vector with dimension " + std::to_string(vector.size()) +
             " was added to vector index \"" + config_.name_ +
             "\", which is configured with dimension " +
             std::to_string(config_.dimensions_) + ".");
  }
  vecSpill_.write(reinterpret_cast<const char*>(vector.data()),
                  static_cast<std::streamsize>(vector.size() * sizeof(float)));
  iriSpill_.write(iri.data(), static_cast<std::streamsize>(iri.size()));
  iriSpill_.put('\n');
  if (!vecSpill_ || !iriSpill_) {
    AD_THROW("Writing to the temporary build files of vector index \"" +
             config_.name_ + "\" failed (disk full?).");
  }
  ids_.push_back(entity.getBits());
  iriOffsets_.push_back(iriSpillOffset_);
  iriLengths_.push_back(static_cast<uint32_t>(iri.size()));
  iriSpillOffset_ += iri.size() + 1;
}

// ____________________________________________________________________________
VectorIndexMetadata VectorIndexBuilder::build() {
  const size_t dim = config_.dimensions_;
  vecSpill_.close();
  iriSpill_.close();
  // The spill files are always removed -- also on success (`dismiss` is only
  // called for the `.tmp` outputs tracked by `outputsCleanup` below).
  FileCleanup spillCleanup;
  spillCleanup.track(vecSpillPath_);
  spillCleanup.track(iriSpillPath_);
  FileCleanup outputsCleanup;

  // 1. Sort by entity id so the reader can binary-search. We sort an index
  //    permutation over the spilled rows. Ties (= duplicate entities) are
  //    broken by insertion order, which both makes the sort deterministic and
  //    lets the dedup below keep the FIRST vector.
  std::vector<size_t> perm(ids_.size());
  std::iota(perm.begin(), perm.end(), size_t{0});
  ql::ranges::sort(perm, [&](size_t a, size_t b) {
    return ids_[a] != ids_[b] ? ids_[a] < ids_[b] : a < b;
  });

  // 2. Deduplicate: duplicate rows in the flat store would yield duplicate
  //    join results. Keep the first vector of each entity and warn (duplicate
  //    input rows are common in ML pipeline output and must not abort a
  //    multi-hour index build). `rows[i]` is the spill index of final row `i`.
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
  perm.clear();
  perm.shrink_to_fit();
  const size_t n = rows.size();

  const size_t numThreads =
      config_.buildThreads_ > 0
          ? config_.buildThreads_
          : std::max(1u, std::thread::hardware_concurrency());

  // All files are first written with a `.tmp` suffix and renamed into place at
  // the very end -- metadata LAST, because the presence of a consistent `.meta`
  // is what makes the loader consider the index. An interrupted or failed
  // build therefore never clobbers a previously working index, and re-running
  // the build always repairs the state.
  const std::string keysPath = vectorKeysFile(basename_, config_.name_);
  const std::string rowmapPath = vectorRowmapFile(basename_, config_.name_);
  const std::string dataPath = vectorDataFile(basename_, config_.name_);
  const std::string irisPath = vectorIrisFile(basename_, config_.name_);
  const std::string hnswPath = vectorHnswFile(basename_, config_.name_);
  const std::string metaPath = vectorMetaFile(basename_, config_.name_);
  auto tmp = [](const std::string& path) { return path + ".tmp"; };

  // 3. Gather the flat store from the spill file in sorted order (parallel
  //    positional reads directly into the memory-mapped destination).
  {
    outputsCleanup.track(tmp(dataPath));
    ad_utility::MmapVector<float> data;
    data.open(n * dim, tmp(dataPath));
    ad_utility::File spill{vecSpillPath_.c_str(), "r"};
    const size_t rowBytes = dim * sizeof(float);
    float* dataBegin = n > 0 ? &data[0] : nullptr;
    parallelOverRows(n, numThreads, [&](size_t, size_t first, size_t last) {
      for (size_t i = first; i < last; ++i) {
        ssize_t read = spill.read(dataBegin + i * dim, rowBytes,
                                  static_cast<off_t>(rows[i] * rowBytes));
        if (read != static_cast<ssize_t>(rowBytes)) {
          throw std::runtime_error{
              "Reading back the temporary vector file failed."};
        }
      }
    });
    // Destructor flushes and writes the trailer.
  }

  // 4. Write the entity mapping: `.keys` (row -> id, ascending by
  //    construction, no tombstones in a fresh build) and `.rowmap` (id -> row).
  {
    outputsCleanup.track(tmp(keysPath));
    outputsCleanup.track(tmp(rowmapPath));
    ad_utility::MmapVector<uint64_t> keys;
    keys.open(n, tmp(keysPath));
    ad_utility::MmapVector<IdRowPair> rowmap;
    rowmap.open(n, tmp(rowmapPath));
    for (size_t i = 0; i < n; ++i) {
      keys[i] = ids_[rows[i]];
      rowmap[i] = IdRowPair{ids_[rows[i]], i};
    }
  }

  // 5. Write the row-aligned `.iris` file (the input for future remaps).
  {
    outputsCleanup.track(tmp(irisPath));
    std::ofstream irisOut{tmp(irisPath), std::ios::binary | std::ios::trunc};
    ad_utility::File spill{iriSpillPath_.c_str(), "r"};
    std::string buffer;
    for (size_t i = 0; i < n; ++i) {
      buffer.resize(iriLengths_[rows[i]]);
      ssize_t read = spill.read(buffer.data(), buffer.size(),
                                static_cast<off_t>(iriOffsets_[rows[i]]));
      if (read != static_cast<ssize_t>(buffer.size())) {
        throw std::runtime_error{"Reading back the temporary IRI file failed."};
      }
      irisOut << buffer << '\n';
    }
    if (!irisOut) {
      AD_THROW("Writing the IRI sidecar of vector index \"" + config_.name_ +
               "\" failed.");
    }
  }

  // 6. Optionally build and save the HNSW graph, keyed by ROW INDEX, with the
  //    vectors read from the just-written memory-mapped flat store: neither
  //    the build nor the resulting file contains a second copy of the vectors.
  bool hasHnsw = config_.buildHnsw_ && n > 0;
  if (hasHnsw) {
    outputsCleanup.track(tmp(hnswPath));
    ad_utility::MmapVectorView<float> data;
    data.open(tmp(dataPath));
    uu::metric_punned_t metric{dim, toUsearchMetric(config_.metric_),
                               uu::scalar_kind_t::f32_k};
    FlatStoreMetric graphMetric{data.data(), dim, metric};
    GraphIndex graph{uu::index_config_t{config_.hnswConnectivity_,
                                        config_.hnswConnectivity_ * 2}};
    uu::index_limits_t limits;
    limits.members = n;
    limits.threads_add = numThreads;
    limits.threads_search = 1;
    if (!graph.try_reserve(limits)) {
      AD_THROW("Could not allocate the HNSW graph for vector index \"" +
               config_.name_ + "\" (" + std::to_string(n) + " nodes).");
    }
    AD_LOG_INFO << "Building the HNSW graph for vector index '" << config_.name_
                << "' (" << n << " vectors, " << numThreads << " threads) ..."
                << std::endl;
    parallelOverRows(n, numThreads, [&](size_t t, size_t first, size_t last) {
      uu::index_update_config_t updateConfig;
      updateConfig.thread = t;
      updateConfig.expansion = config_.hnswExpansionAdd_;
      for (size_t row = first; row < last; ++row) {
        auto added =
            graph.add(row, graphMetric.rowPtr(row), graphMetric, updateConfig);
        if (!added) {
          throw std::runtime_error{
              std::string{"Could not add a vector to the HNSW graph: "} +
              added.error.what()};
        }
      }
    });
    std::ofstream hnswOut{tmp(hnswPath), std::ios::binary | std::ios::trunc};
    auto saved = graph.save_to_stream([&](const void* buffer, size_t length) {
      hnswOut.write(reinterpret_cast<const char*>(buffer),
                    static_cast<std::streamsize>(length));
      return static_cast<bool>(hnswOut);
    });
    if (!saved || !hnswOut) {
      AD_THROW("Could not save the HNSW graph of vector index \"" +
               config_.name_ + "\": " + saved.error.what());
    }
  }

  // 7. Write the metadata file.
  VectorIndexMetadata meta;
  meta.config_ = config_;
  meta.numVectors_ = n;
  meta.numTombstones_ = 0;
  meta.hasHnsw_ = hasHnsw;
  meta.version_ = VECTOR_INDEX_VERSION;
  meta.vocabSize_ = vocabSize_;
  {
    outputsCleanup.track(tmp(metaPath));
    std::ofstream metaOut{tmp(metaPath)};
    if (!metaOut.is_open()) {
      AD_THROW("Could not write the vector metadata file " + tmp(metaPath));
    }
    metaOut << nlohmann::json(meta).dump(2);
  }

  // 8. Rename everything into place (metadata last, see above). A stale
  //    `.hnsw` file from a previous build is removed if this build has none.
  std::filesystem::rename(tmp(dataPath), dataPath);
  std::filesystem::rename(tmp(keysPath), keysPath);
  std::filesystem::rename(tmp(rowmapPath), rowmapPath);
  std::filesystem::rename(tmp(irisPath), irisPath);
  if (hasHnsw) {
    std::filesystem::rename(tmp(hnswPath), hnswPath);
  } else {
    std::error_code ec;
    std::filesystem::remove(hnswPath, ec);
  }
  std::filesystem::rename(tmp(metaPath), metaPath);
  outputsCleanup.dismiss();
  return meta;
}

}  // namespace qlever::vector
