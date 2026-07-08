// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndexBuilder.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <filesystem>
#include <mutex>
#include <numeric>
#include <thread>

#include "backports/algorithm.h"
#include "services/vectorSearch/UsearchGraph.h"
#include "services/vectorSearch/VectorMemory.h"
#include "util/Exception.h"
#include "util/File.h"
#include "util/Log.h"
#include "util/MmapVector.h"
#include "util/jthread.h"

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

// Run `job(threadIdx, firstRow, lastRow, shouldStop)` for a partition of
// `[0, n)` over `numThreads` threads and rethrow the first error, if any.
// `shouldStop` becomes true as soon as any thread fails, so a long job (a
// multi-hour HNSW build) can abort promptly instead of running to completion.
template <typename Job>
void parallelOverRows(size_t n, size_t numThreads, const Job& job) {
  numThreads = std::max<size_t>(1, std::min(numThreads, n));
  std::atomic<bool> stop{false};
  std::mutex errorMutex;
  std::string firstError;
  auto recordError = [&](std::string message) {
    stop.store(true, std::memory_order_relaxed);
    std::lock_guard lock{errorMutex};
    if (firstError.empty()) {
      firstError =
          message.empty() ? std::string{"unknown build error"} : message;
    }
  };
  auto guardedJob = [&](size_t t, size_t first, size_t last) {
    try {
      job(t, first, last, stop);
    } catch (const std::exception& e) {
      recordError(e.what());
    } catch (...) {
      recordError("non-standard exception");
    }
  };
  if (numThreads == 1) {
    guardedJob(0, 0, n);
  } else {
    std::vector<ad_utility::JThread> threads;
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

// Above this store size the O(n^2) brute-force r(d) fallback is refused; the
// build then needs `hnsw: true` (the self-kNN searches the graph) or a
// precomputed `cslsR` sidecar (the GPU path).
constexpr size_t CSLS_BRUTE_FORCE_MAX = 50'000;

// The FINE layer of the just-written flat store(s): what the csls r(d)
// cosine distances are computed on -- the rerank matrix of a two-layer build,
// else the single scan store -- matching the query-time exact baseline, which
// reads the same layer.
struct CslsFineLayer {
  ad_utility::MmapVectorView<char> data_;
  uu::metric_punned_t metric_;
  size_t rowBytes_;
  const char* ptr(size_t row) const { return data_.data() + row * rowBytes_; }
  float distance(size_t a, size_t b) const {
    return static_cast<float>(
        metric_(reinterpret_cast<const uu::byte_t*>(ptr(a)),
                reinterpret_cast<const uu::byte_t*>(ptr(b))));
  }
};

// Log the r(d) distribution, and WARN when the embedding space is
// near-saturated (median cosine similarity to the nearest neighbours >= 0.95:
// cos ~ 1 everywhere makes CSLS ~ 0 for every candidate, so the threshold cut
// carries almost no signal -- the documented CSLS failure mode).
void logCslsDistribution(const std::string& indexName,
                         const std::vector<float>& r) {
  if (r.empty()) {
    return;
  }
  std::vector<float> sorted = r;
  ql::ranges::sort(sorted);
  const size_t n = sorted.size();
  const float min = sorted.front();
  const float p50 = sorted[n / 2];
  const float p95 = sorted[std::min(n - 1, (n * 95) / 100)];
  const float max = sorted.back();
  AD_LOG_INFO << "Vector index \"" << indexName
              << "\": csls r(d): min/p50/p95/max = " << min << "/" << p50 << "/"
              << p95 << "/" << max << std::endl;
  if (p50 >= 0.95f) {
    AD_LOG_WARN
        << "Vector index \"" << indexName
        << "\": the embedding space is near-saturated (median r(d) = " << p50
        << " >= 0.95, i.e. cosine similarity ~ 1 everywhere). CSLS ~ 0 for "
           "every candidate then, so a `vec:cslsThreshold` cut carries "
           "little signal on this index."
        << std::endl;
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
  if (config_.dimensions_ > MAX_VECTOR_DIMENSIONS) {
    AD_THROW("Vector index \"" + config_.name_ +
             "\" has an implausible "
             "dimension (" +
             std::to_string(config_.dimensions_) + ").");
  }
  // Raw row byte length via `rowBytesFor`: `dim * bytesPerScalar`, except for
  // the sign-packed `binary` scalar whose rows are `(dim + 7) / 8` bytes
  // (usearch's f32 -> b1x8 cast packs bit i of a row iff component i > 0).
  rowBytes_ = rowBytesFor(config_.scalar_, config_.dimensions_);
  fromF32_ =
      uu::casts_punned_t::make(toUsearchScalar(config_.scalar_)).from.f32;
  castBuffer_.resize(rowBytes_);
  vecSpillPath_ = vectorDataFile(basename_, config_.name_) + ".spill";
  iriSpillPath_ = vectorIrisFile(basename_, config_.name_) + ".spill";
  vecSpill_.open(vecSpillPath_, std::ios::binary | std::ios::trunc);
  iriSpill_.open(iriSpillPath_, std::ios::binary | std::ios::trunc);
  if (!vecSpill_.is_open() || !iriSpill_.is_open()) {
    AD_THROW("Could not create temporary build files for vector index \"" +
             config_.name_ + "\" next to " + basename_);
  }
  // CSLS is cosine-specific: the query-time cut converts the stored distance
  // back to a similarity as `cos_sim = 1 - distance`, which only holds for
  // the cosine metric (validated upstream in `parseSpec`; checked again here
  // for direct builder users).
  if (config_.csls_ && config_.metric_ != VectorMetric::Cosine) {
    AD_THROW("Vector index \"" + config_.name_ +
             "\": `csls` works in cosine-similarity space, so it requires "
             "`metric: cosine` (got `" +
             toString(config_.metric_) + "`).");
  }
  // A `binary` store without a rerank layer serves only integer HAMMING
  // distances -- there is no exact cosine anywhere to compute r(d) or the
  // query-time CSLS from.
  if (config_.csls_ && config_.scalar_ == VectorScalar::Binary &&
      !config_.rerankScalar_.has_value()) {
    AD_THROW("Vector index \"" + config_.name_ +
             "\": `csls` needs exact cosine distances, but a `binary` store "
             "without a `rerank` layer only serves Hamming distances. Add "
             "e.g. `\"rerank\": \"bf16\"`.");
  }
  // The optional fine rerank layer: a second, independently encoded spill of
  // the SAME input rows (validated i8/binary-free upstream in `parseSpec`;
  // checked again here so direct builder users cannot request a quantized
  // rerank layer).
  if (config_.rerankScalar_.has_value()) {
    if (config_.rerankScalar_.value() == VectorScalar::I8 ||
        config_.rerankScalar_.value() == VectorScalar::Binary) {
      AD_THROW("The rerank layer of vector index \"" + config_.name_ +
               "\" must be a high-precision scalar (bf16, f16, or f32), not " +
               toString(config_.rerankScalar_.value()) + ".");
    }
    rerankRowBytes_ =
        rowBytesFor(config_.rerankScalar_.value(), config_.dimensions_);
    rerankFromF32_ =
        uu::casts_punned_t::make(toUsearchScalar(config_.rerankScalar_.value()))
            .from.f32;
    rerankCastBuffer_.resize(rerankRowBytes_);
    rerankSpillPath_ =
        vectorRerankDataFile(basename_, config_.name_) + ".spill";
    rerankSpill_.open(rerankSpillPath_, std::ios::binary | std::ios::trunc);
    if (!rerankSpill_.is_open()) {
      AD_THROW("Could not create temporary build files for vector index \"" +
               config_.name_ + "\" next to " + basename_);
    }
  }
}

// ____________________________________________________________________________
VectorIndexBuilder::~VectorIndexBuilder() {
  // Idempotent: `build()` already removes these on success (and its
  // `FileCleanup` removes them on a throw inside `build()`); this covers a
  // builder destroyed before `build()` was called.
  std::error_code ec;
  std::filesystem::remove(vecSpillPath_, ec);
  std::filesystem::remove(iriSpillPath_, ec);
  if (!rerankSpillPath_.empty()) {
    std::filesystem::remove(rerankSpillPath_, ec);
  }
}

// ____________________________________________________________________________
void VectorIndexBuilder::add(Id entity, std::string_view iri,
                             ql::span<const float> vector,
                             std::optional<float> cslsR) {
  if (vector.size() != config_.dimensions_) {
    AD_THROW("A vector with dimension " + std::to_string(vector.size()) +
             " was added to vector index \"" + config_.name_ +
             "\", which is configured with dimension " +
             std::to_string(config_.dimensions_) + ".");
  }
  // Ingested r(d) values are all-or-nothing: a row with a value while earlier
  // rows had none (or vice versa) would silently misalign the sidecar.
  if (cslsR.has_value()) {
    if (!config_.csls_) {
      AD_THROW("A cslsR value was added to vector index \"" + config_.name_ +
               "\", which is not configured with `csls: true`.");
    }
    if (cslsRInput_.size() != ids_.size()) {
      AD_THROW("Vector index \"" + config_.name_ +
               "\": some rows carry a cslsR value and some do not; the "
               "precomputed r(d) must cover every row.");
    }
    if (!std::isfinite(cslsR.value())) {
      AD_THROW("Vector index \"" + config_.name_ + "\": row " +
               std::to_string(ids_.size() + 1) +
               " has a non-finite cslsR value.");
    }
    cslsRInput_.push_back(cslsR.value());
  } else if (!cslsRInput_.empty()) {
    AD_THROW("Vector index \"" + config_.name_ +
             "\": some rows carry a cslsR value and some do not; the "
             "precomputed r(d) must cover every row.");
  }
  // Convert to the storage scalar (no-op for f32). The `binary` sign-pack
  // cast (usearch's f32 -> b1x8) memsets only the `dim / 8` WHOLE output
  // bytes and then ORs bits in, so a partial trailing byte would keep stale
  // bits of the PREVIOUS row in the reused buffer -- zero it first.
  const char* rowBytesPtr = reinterpret_cast<const char*>(vector.data());
  if (config_.scalar_ == VectorScalar::Binary) {
    ql::ranges::fill(castBuffer_, char{0});
  }
  if (fromF32_ != nullptr &&
      fromF32_(rowBytesPtr, config_.dimensions_, castBuffer_.data())) {
    rowBytesPtr = castBuffer_.data();
  }
  vecSpill_.write(rowBytesPtr, static_cast<std::streamsize>(rowBytes_));
  // Two-layer build: also spill the SAME f32 row at the rerank precision (both
  // layers are encoded from the original input, never from each other).
  if (config_.rerankScalar_.has_value()) {
    const char* rerankPtr = reinterpret_cast<const char*>(vector.data());
    if (rerankFromF32_ != nullptr &&
        rerankFromF32_(rerankPtr, config_.dimensions_,
                       rerankCastBuffer_.data())) {
      rerankPtr = rerankCastBuffer_.data();
    }
    rerankSpill_.write(rerankPtr,
                       static_cast<std::streamsize>(rerankRowBytes_));
    if (!rerankSpill_) {
      AD_THROW("Writing to the temporary build files of vector index \"" +
               config_.name_ + "\" failed (disk full?).");
    }
  }
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
  const bool twoLayer = config_.rerankScalar_.has_value();
  vecSpill_.close();
  iriSpill_.close();
  if (twoLayer) {
    rerankSpill_.close();
  }
  // A failed flush at close (e.g. disk full) would leave a truncated spill;
  // the gather below reads by absolute offset and (because `File::read` loops
  // to EOF) would otherwise hang instead of erroring. Verify the spill sizes.
  if (vecSpill_.fail() || iriSpill_.fail() ||
      (twoLayer && rerankSpill_.fail())) {
    AD_THROW("Writing the temporary build files of vector index \"" +
             config_.name_ + "\" failed (disk full?).");
  }
  {
    std::error_code ec;
    auto vecSize = std::filesystem::file_size(vecSpillPath_, ec);
    if (ec || vecSize != ids_.size() * rowBytes_) {
      AD_THROW("The temporary vector file of vector index \"" + config_.name_ +
               "\" is incomplete (disk full?).");
    }
    auto iriSize = std::filesystem::file_size(iriSpillPath_, ec);
    if (ec || iriSize != iriSpillOffset_) {
      AD_THROW("The temporary IRI file of vector index \"" + config_.name_ +
               "\" is incomplete (disk full?).");
    }
    if (twoLayer) {
      auto rerankSize = std::filesystem::file_size(rerankSpillPath_, ec);
      if (ec || rerankSize != ids_.size() * rerankRowBytes_) {
        AD_THROW("The temporary rerank file of vector index \"" +
                 config_.name_ + "\" is incomplete (disk full?).");
      }
    }
  }
  // The spill files are always removed -- also on success (`dismiss` is only
  // called for the `.tmp` outputs tracked by `outputsCleanup` below).
  FileCleanup spillCleanup;
  spillCleanup.track(vecSpillPath_);
  spillCleanup.track(iriSpillPath_);
  if (twoLayer) {
    spillCleanup.track(rerankSpillPath_);
  }
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

  // Byte stride between consecutive rows in the flat store: the natural row
  // byte length. Real embedding dimensions are multiples of 64, so this is
  // already 64-byte aligned; the NumKong distance kernels use unaligned loads
  // with masked tails regardless, so alignment never affects correctness.
  const size_t stride = rowBytes_;

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
  const std::string rerankPath = vectorRerankDataFile(basename_, config_.name_);
  const std::string irisPath = vectorIrisFile(basename_, config_.name_);
  const std::string hnswPath = vectorHnswFile(basename_, config_.name_);
  const std::string cslsPath = vectorCslsFile(basename_, config_.name_);
  const std::string metaPath = vectorMetaFile(basename_, config_.name_);
  auto tmp = [](const std::string& path) { return path + ".tmp"; };

  // Open the FINE layer of the just-gathered store(s) for the csls r(d)
  // computation (see `CslsFineLayer`). Only called after step 3 has written
  // the matrices.
  auto openFineLayer = [&]() {
    CslsFineLayer fine;
    fine.data_.open(twoLayer ? tmp(rerankPath) : tmp(dataPath),
                    ad_utility::AccessPattern::Random);
    fine.rowBytes_ = twoLayer ? rerankRowBytes_ : rowBytes_;
    const VectorScalar fineScalar = config_.rerankScalar_.value_or(
        config_.scalar_);  // never `binary` with csls (rejected in the ctor)
    fine.metric_ = uu::metric_punned_t{config_.dimensions_,
                                       toUsearchMetric(config_.metric_),
                                       toUsearchScalar(fineScalar)};
    return fine;
  };

  // Gather one flat matrix from its spill file in sorted `rows` order
  // (parallel positional reads directly into the memory-mapped destination).
  // Used for the scan `.data` and -- with the rerank row length -- the
  // optional fine `.rerank.data`; both use the natural (unpadded) row stride.
  auto gatherMatrix = [&](const std::string& outPath,
                          const std::string& spillPath, size_t layerRowBytes) {
    outputsCleanup.track(tmp(outPath));
    ad_utility::MmapVector<char> data;
    data.open(n * layerRowBytes, tmp(outPath));
    ad_utility::File spill{spillPath.c_str(), "r"};
    char* dataBegin = n > 0 ? &data[0] : nullptr;
    parallelOverRows(
        n, numThreads, [&](size_t, size_t first, size_t last, auto& stop) {
          for (size_t i = first; i < last; ++i) {
            if (stop.load(std::memory_order_relaxed)) return;
            ssize_t read =
                spill.read(dataBegin + i * layerRowBytes, layerRowBytes,
                           static_cast<off_t>(rows[i] * layerRowBytes));
            if (read != static_cast<ssize_t>(layerRowBytes)) {
              throw std::runtime_error{
                  "Reading back the temporary vector file failed."};
            }
          }
        });
    // Destructor flushes and writes the trailer.
  };

  // 3. Gather the flat scan store (and, for a two-layer build, the fine
  //    rerank store: the SAME rows in the SAME order at the rerank precision).
  gatherMatrix(dataPath, vecSpillPath_, rowBytes_);
  if (twoLayer) {
    gatherMatrix(rerankPath, rerankSpillPath_, rerankRowBytes_);
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
    irisOut.close();
    if (irisOut.fail()) {
      AD_THROW("Writing the IRI sidecar of vector index \"" + config_.name_ +
               "\" failed (disk full?).");
    }
  }

  // The spill files are no longer needed (the flat store(s) and the `.iris`
  // file have been written). Remove them NOW, before the multi-hour HNSW
  // build, so the transient disk footprint is ~1x the vector matrix rather
  // than ~2x.
  {
    std::error_code ec;
    std::filesystem::remove(vecSpillPath_, ec);
    std::filesystem::remove(iriSpillPath_, ec);
    if (twoLayer) {
      std::filesystem::remove(rerankSpillPath_, ec);
    }
  }

  // 6. Optionally build and save the HNSW graph, keyed by ROW INDEX, with the
  //    vectors read from the just-written memory-mapped flat store: neither
  //    the build nor the resulting file contains a second copy of the vectors.
  bool hasHnsw = config_.buildHnsw_ && n > 0;
  // The per-final-row csls r(d) (filled iff `config_.csls_`): ingested from
  // `add`'s cslsR values, or computed by a self-kNN -- against the in-RAM
  // HNSW graph below when there is one, else by the (small-store) brute
  // force of step 6b.
  std::vector<float> cslsR;
  const bool cslsSelfKnn = config_.csls_ && cslsRInput_.empty();
  if (hasHnsw) {
    outputsCleanup.track(tmp(hnswPath));
    ad_utility::MmapVectorView<char> data;
    // Random access: graph construction reads vectors in graph order, not
    // sequentially, so read-ahead only pollutes the page cache.
    data.open(tmp(dataPath), ad_utility::AccessPattern::Random);
    // Warn if the flat store cannot plausibly stay page-cache-resident: HNSW
    // construction random-accesses it, so once it exceeds RAM the build
    // becomes disk-bound.
    if (uint64_t ram = totalPhysicalMemoryBytes();
        ram != 0 && n * stride > ram) {
      AD_LOG_WARN << "Vector index \"" << config_.name_ << "\": the "
                  << (n * stride >> 30) << " GiB flat store exceeds physical "
                  << "memory (" << (ram >> 30)
                  << " GiB); the HNSW build will be disk-bound. Consider a "
                     "smaller scalar type (f16/i8) or more RAM."
                  << std::endl;
    }
    // The graph is built over the SCAN layer, so it uses the scan metric:
    // HAMMING over the sign bits for a `binary` store (the reader's
    // `graphMetric()` mirrors this), the index metric otherwise.
    uu::metric_punned_t metric{
        config_.dimensions_,
        toUsearchScanMetric(config_.scalar_, config_.metric_),
        toUsearchScalar(config_.scalar_)};
    FlatStoreMetric graphMetric{data.data(), stride, n, metric};
    GraphIndex graph{uu::index_config_t{config_.hnswConnectivity_,
                                        config_.hnswConnectivity_ * 2}};
    uu::index_limits_t limits;
    limits.members = n;
    limits.threads_add = numThreads;
    // The csls self-kNN below searches the graph from all build threads.
    limits.threads_search = cslsSelfKnn ? numThreads : 1;
    if (!graph.try_reserve(limits)) {
      AD_THROW("Could not allocate the HNSW graph for vector index \"" +
               config_.name_ + "\" (" + std::to_string(n) + " nodes).");
    }
    AD_LOG_INFO << "Building the HNSW graph for vector index '" << config_.name_
                << "' (" << n << " vectors, " << numThreads << " threads) ..."
                << std::endl;
    parallelOverRows(
        n, numThreads, [&](size_t t, size_t first, size_t last, auto& stop) {
          uu::index_update_config_t updateConfig;
          updateConfig.thread = t;
          updateConfig.expansion = config_.hnswExpansionAdd_;
          for (size_t row = first; row < last; ++row) {
            if (stop.load(std::memory_order_relaxed)) return;
            auto added = graph.add(row, graphMetric.rowPtr(row), graphMetric,
                                   updateConfig);
            if (!added) {
              throw std::runtime_error{
                  std::string{"Could not add a vector to the HNSW graph: "} +
                  added.error.what()};
            }
          }
        });
    if (cslsSelfKnn) {
      // csls r(d) via the in-RAM graph: search every row's OWN stored vector,
      // drop the self hit (by row identity), and average the cosine
      // similarity `1 - distance` of the top-`cslsNeighbors` neighbours ON
      // THE FINE LAYER. The graph ranks by the COARSE scan metric (i8 cosine
      // or binary Hamming on a two-layer build), so a two-layer build
      // over-fetches with the SERVICE's rerank margin (`defaultRerankK`) and
      // re-scores the candidates exactly on the fine layer; a single-layer
      // graph already ranks in the fine metric, so `k + 1` (the self hit)
      // suffices.
      const CslsFineLayer fine = openFineLayer();
      const size_t k = config_.cslsNeighbors_;
      const size_t fetch = std::min(
          n, twoLayer ? std::max(defaultRerankK(config_.scalar_, k), k + 1)
                      : k + 1);
      // RECALL-tuned search expansion (efSearch): the self-kNN feeds a
      // persisted statistic, so it uses `max(200, 20 * k, fetch)` rather than
      // the (typically much smaller) query-time `hnswExpansionSearch_`.
      const size_t expansion = std::max({size_t{200}, 20 * k, fetch});
      AD_LOG_INFO << "Vector index \"" << config_.name_
                  << "\": computing csls r(d) via the HNSW graph (" << n
                  << " vectors, " << k << " neighbours, expansion " << expansion
                  << ") ..." << std::endl;
      cslsR.assign(n, 0.f);
      parallelOverRows(
          n, numThreads, [&](size_t t, size_t first, size_t last, auto& stop) {
            std::vector<uint64_t> cand(fetch);
            std::vector<uu::distance_punned_t> coarseDists(fetch);
            for (size_t row = first; row < last; ++row) {
              if (stop.load(std::memory_order_relaxed)) return;
              uu::index_search_config_t searchConfig;
              searchConfig.thread = t;
              searchConfig.expansion = expansion;
              auto found = graph.search(graphMetric.rowPtr(row), fetch,
                                        graphMetric, searchConfig);
              if (!found) {
                throw std::runtime_error{
                    std::string{"The csls r(d) self-search on the HNSW graph "
                                "failed: "} +
                    found.error.what()};
              }
              size_t count = found.dump_to(cand.data(), coarseDists.data());
              CslsNeighborhood top{k};
              for (size_t i = 0; i < count; ++i) {
                if (cand[i] == row) {
                  continue;  // SELF-EXCLUDED, by row identity
                }
                top.offer(fine.distance(row, cand[i]));
              }
              cslsR[row] = top.meanCosSim();
            }
          });
    }
    std::ofstream hnswOut{tmp(hnswPath), std::ios::binary | std::ios::trunc};
    bool streamOk = true;
    auto saved = graph.save_to_stream([&](const void* buffer, size_t length) {
      hnswOut.write(reinterpret_cast<const char*>(buffer),
                    static_cast<std::streamsize>(length));
      streamOk = static_cast<bool>(hnswOut);
      return streamOk;
    });
    hnswOut.close();
    if (!saved || !streamOk || hnswOut.fail()) {
      AD_THROW("Could not save the HNSW graph of vector index \"" +
               config_.name_ + "\" (disk full?): " +
               (saved ? "stream write failed" : saved.error.what()));
    }
  }

  // 6b. The csls r(d) sidecar. The HNSW self-kNN above already filled `cslsR`
  //     when there was a graph; otherwise the values are ingested verbatim
  //     (the precomputed `cslsR` GPU path) or brute-forced for a small store.
  if (config_.csls_) {
    if (!cslsRInput_.empty()) {
      // Ingested: `add` enforced one value per row, so the permutation below
      // is total. Carried verbatim through the same sort/dedup as the rows.
      AD_CORRECTNESS_CHECK(cslsRInput_.size() == ids_.size());
      cslsR.resize(n);
      for (size_t i = 0; i < n; ++i) {
        cslsR[i] = cslsRInput_[rows[i]];
      }
    } else if (!hasHnsw) {
      if (n >= CSLS_BRUTE_FORCE_MAX) {
        AD_THROW("Vector index \"" + config_.name_ +
                 "\": csls needs `hnsw: true` or a precomputed `cslsR` for "
                 "indexes with >= " +
                 std::to_string(CSLS_BRUTE_FORCE_MAX) +
                 " vectors (the brute-force r(d) fallback is O(n^2); got " +
                 std::to_string(n) + " vectors).");
      }
      // Small store: EXACT r(d) by brute force over the fine layer.
      const CslsFineLayer fine = openFineLayer();
      const size_t k = config_.cslsNeighbors_;
      cslsR.assign(n, 0.f);
      parallelOverRows(n, numThreads,
                       [&](size_t, size_t first, size_t last, auto& stop) {
                         for (size_t row = first; row < last; ++row) {
                           if (stop.load(std::memory_order_relaxed)) return;
                           CslsNeighborhood top{k};
                           for (size_t j = 0; j < n; ++j) {
                             if (j == row) {
                               continue;  // SELF-EXCLUDED, by row identity
                             }
                             top.offer(fine.distance(row, j));
                           }
                           cslsR[row] = top.meanCosSim();
                         }
                       });
    }
    AD_CORRECTNESS_CHECK(cslsR.size() == n);
    logCslsDistribution(config_.name_, cslsR);
    {
      outputsCleanup.track(tmp(cslsPath));
      ad_utility::MmapVector<float> out;
      out.open(n, tmp(cslsPath));
      ql::ranges::copy(cslsR, out.begin());
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
  meta.rowStrideBytes_ = stride;
  meta.collationLocale_ = collationLocale_;
  {
    outputsCleanup.track(tmp(metaPath));
    std::ofstream metaOut{tmp(metaPath)};
    if (!metaOut.is_open()) {
      AD_THROW("Could not write the vector metadata file " + tmp(metaPath));
    }
    metaOut << nlohmann::json(meta).dump(2);
    metaOut.close();
    if (metaOut.fail()) {
      AD_THROW("Could not write the vector metadata file " + tmp(metaPath) +
               " (disk full?).");
    }
  }

  // 8. Rename everything into place (metadata last, see above). A stale
  //    `.hnsw`/`.rerank.data`/`.csls` file from a previous build is removed
  //    if this build has none.
  std::filesystem::rename(tmp(dataPath), dataPath);
  if (twoLayer) {
    std::filesystem::rename(tmp(rerankPath), rerankPath);
  } else {
    std::error_code ec;
    std::filesystem::remove(rerankPath, ec);
  }
  std::filesystem::rename(tmp(keysPath), keysPath);
  std::filesystem::rename(tmp(rowmapPath), rowmapPath);
  std::filesystem::rename(tmp(irisPath), irisPath);
  if (hasHnsw) {
    std::filesystem::rename(tmp(hnswPath), hnswPath);
  } else {
    std::error_code ec;
    std::filesystem::remove(hnswPath, ec);
  }
  if (config_.csls_) {
    std::filesystem::rename(tmp(cslsPath), cslsPath);
  } else {
    std::error_code ec;
    std::filesystem::remove(cslsPath, ec);
  }
  std::filesystem::rename(tmp(metaPath), metaPath);
  outputsCleanup.dismiss();
  return meta;
}

}  // namespace qlever::vector
