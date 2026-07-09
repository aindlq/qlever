// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndex.h"

#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "backports/StartsWithAndEndsWith.h"
#include "backports/algorithm.h"
#include "services/vectorSearch/UsearchGraph.h"
#include "services/vectorSearch/VectorMemory.h"
#include "util/Exception.h"
#include "util/HashSet.h"
#include "util/Log.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
// How often the exact-search loops call the interrupt callback. Frequent enough
// for sub-second cancellation latency, rare enough to not show up in profiles.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// Hard upper bound on the number of results a single search returns, regardless
// of the user's `k` and the index size. `k` is remote and unbounded; without
// this cap a query like `vec:k 100000000` on a 100M-vector index would drive
// gigabytes of (unaccounted) scratch and an effectively whole-index HNSW probe.
// A nearest-neighbour query asking for more than this is pathological; the
// result is capped (not an error). 100k results = ~1.2 MB of key+distance
// buffers, and a bounded HNSW search.
constexpr size_t MAX_SEARCH_RESULTS = 100'000;

// A pool of usearch search-context ids. Each concurrent HNSW search needs a
// distinct context; a request beyond the pool size briefly WAITS for a free
// context instead of failing the query (the pool is sized generously above the
// hardware concurrency, so waiting is rare).
class SearchSlotPool {
 public:
  explicit SearchSlotPool(size_t numSlots) {
    for (size_t i = 0; i < numSlots; ++i) {
      free_.push(i);
    }
  }

  class Slot {
   public:
    Slot(SearchSlotPool& pool, size_t id) : pool_{&pool}, id_{id} {}
    Slot(const Slot&) = delete;
    Slot& operator=(const Slot&) = delete;
    ~Slot() { pool_->release(id_); }
    size_t id() const { return id_; }

   private:
    SearchSlotPool* pool_;
    size_t id_;
  };

  // Acquire a slot, waiting if none is free. `checkInterrupt`, if set, is
  // invoked while waiting so that a cancelled/timed-out query does not park a
  // worker thread here indefinitely (it throws to unwind).
  Slot acquire(const CheckInterruptCallback& checkInterrupt) {
    std::unique_lock lock{mutex_};
    while (free_.empty()) {
      if (checkInterrupt) {
        cv_.wait_for(lock, std::chrono::milliseconds{50},
                     [this] { return !free_.empty(); });
        if (free_.empty()) {
          lock.unlock();
          checkInterrupt();  // throws on cancellation
          lock.lock();
        }
      } else {
        cv_.wait(lock, [this] { return !free_.empty(); });
      }
    }
    size_t id = free_.front();
    free_.pop();
    return Slot{*this, id};
  }

 private:
  void release(size_t id) {
    {
      std::lock_guard lock{mutex_};
      free_.push(id);
    }
    cv_.notify_one();
  }

  std::mutex mutex_;
  std::condition_variable cv_;
  std::queue<size_t> free_;
};
}  // namespace

// Deleter for the `posix_memalign`-allocated aligned copy (freed with
// `std::free`, NOT `delete`).
struct AlignedFree {
  void operator()(void* p) const { std::free(p); }
};

// All heavy members (mmaped files + usearch) live here so that neither usearch
// nor `MmapVector` leak into the public header.
struct VectorIndex::Impl {
  // One flat row-major matrix with its metric, f32 casts, and RAM-residency
  // state. Every index has the SCAN layer (`.data`, the configured `scalar_`);
  // a two-layer index additionally has the fine RERANK layer (`.rerank.data`).
  // Both share the entity mappings (`keys_`/`rowmap_`) and the row order.
  struct MatrixLayer {
    ad_utility::MmapVectorView<char> data_;
    std::optional<uu::metric_punned_t> metric_;
    uu::casts_punned_t casts_;  // f32 <-> this layer's storage scalar
    size_t dim_ = 0;
    // Raw (unpadded) per-row byte length of this layer.
    size_t rowBytes_ = 0;
    // Byte stride between consecutive rows (>= `rowBytes_`). Derived from the
    // metadata at open; for a `Residency::AlignedCopy` it becomes the padded
    // stride of `alignedBuf_`.
    size_t stride_ = 0;
    // Optional 64-byte-aligned RAM copy of the whole matrix (`Residency`
    // `AlignedCopy`). When set, `base()` reads from it (with the padded
    // `stride_`) instead of the memory-mapped file.
    std::unique_ptr<char, AlignedFree> alignedBuf_;
    // The residency strategy last applied (best-effort; see
    // `VectorIndex::residency()`). Stays `None` when the request was skipped
    // by the fits-in-RAM gate.
    Residency residency_ = Residency::None;

    // Base pointer of the active matrix: the aligned RAM copy if present,
    // else the memory-mapped file.
    const char* base() const {
      return alignedBuf_ ? alignedBuf_.get() : data_.data();
    }

    // Pointer to the start of row `i` in the active matrix (storage scalar
    // bytes). Uses the per-row stride, which may exceed `rowBytes_` when rows
    // are padded for SIMD alignment; the metric reads only `dim_` scalars.
    const char* rowPtr(size_t i) const { return base() + i * stride_; }

    // Encode an f32 query into this layer's storage scalar. Returns a pointer
    // to the encoded bytes; `buffer` provides the storage when a conversion
    // happens. The buffer is ZEROED first (`assign`, not `resize`): usearch's
    // f32 -> b1x8 sign-pack cast memsets only the `dim / 8` WHOLE output
    // bytes and then ORs bits in, so a partial trailing byte of a reused
    // buffer would keep stale bits (all other casts overwrite every byte, for
    // which the zeroing is merely redundant).
    const char* encodeQuery(ql::span<const float> query,
                            std::vector<char>& buffer) const {
      const char* raw = reinterpret_cast<const char*>(query.data());
      auto fromF32 = casts_.from.f32;
      if (fromF32 != nullptr) {
        buffer.assign(rowBytes_, 0);
        if (fromF32(raw, dim_, buffer.data())) {
          return buffer.data();
        }
      }
      return raw;
    }

    // Distance between two (encoded, `rowBytes_`-sized) vectors, using the
    // (punned) layer metric so that exact and HNSW distances are identical.
    float distanceBetweenBytes(const char* a, const char* b) const {
      return static_cast<float>(
          metric_.value()(reinterpret_cast<const uu::byte_t*>(a),
                          reinterpret_cast<const uu::byte_t*>(b)));
    }

    // Distance between an (encoded) query and row `i`.
    float distanceToRow(const char* queryBytes, size_t i) const {
      return distanceBetweenBytes(queryBytes, rowPtr(i));
    }
  };

  VectorIndexMetadata meta_;
  ad_utility::MmapVectorView<uint64_t> keys_;  // row -> id (or TOMBSTONE_KEY)
  ad_utility::MmapVectorView<IdRowPair> rowmap_;  // id -> row, sorted by id
  // The coarse scan matrix `.data` (the only matrix of a single-layer index).
  MatrixLayer scan_;
  // The fine rerank matrix `.rerank.data`, iff `meta_.config_.rerankScalar_`.
  std::optional<MatrixLayer> rerank_;
  // The row-aligned csls hubness sidecar `.csls` (one f32 r(d) per row), iff
  // `meta_.config_.csls_`.
  ad_utility::MmapVectorView<float> cslsR_;
  std::optional<GraphIndex> graph_;  // present iff meta_.hasHnsw_
  std::unique_ptr<SearchSlotPool> searchSlots_;
  // Sticky safety net for the sequential-read hint of the merge-join gather.
  // The gather emits rows in non-decreasing order iff the store is genuinely
  // id-sorted (the normal case). If a gather is ever observed to emit rows out
  // of order (a changed KG collation, so the store's layout no longer matches
  // the current ids), this latches and later gathers stop advising SEQUENTIAL
  // (which would mislead the kernel's read-ahead). Correctness is unaffected
  // either way; this only tunes the `madvise` hint. `mutable` so the const
  // search methods can set it; `atomic` for concurrent searches.
  mutable std::atomic<bool> gatherNonMonotonic_{false};

  size_t dim() const { return meta_.config_.dimensions_; }
  size_t numLive() const { return meta_.numVectors_ - meta_.numTombstones_; }

  // The FINE layer: what the exact primitives (`vec:distance`,
  // `searchExact*`, `getVector`) score on -- the rerank matrix when present,
  // else the single scan store. (The non-const overload exists solely for the
  // advisory `madvise` hints of the exact scans.)
  const MatrixLayer& fine() const {
    return rerank_.has_value() ? rerank_.value() : scan_;
  }
  MatrixLayer& fine() { return rerank_.has_value() ? rerank_.value() : scan_; }

  // Row index of `entity`, or nullopt if it has no (live) vector here.
  std::optional<size_t> rowOf(Id entity) const {
    uint64_t id = entity.getBits();
    auto it = ql::ranges::lower_bound(
        rowmap_, id, std::less<>{},
        [](const IdRowPair& pair) { return pair.idBits_; });
    if (it == rowmap_.end() || it->idBits_ != id) {
      return std::nullopt;
    }
    return static_cast<size_t>(it->row_);
  }

  // The HNSW graph reads the SCAN layer (the graph is built on it, so its
  // distances live in the coarse space of a two-layer index).
  FlatStoreMetric graphMetric() const {
    return FlatStoreMetric{scan_.base(), scan_.stride_, meta_.numVectors_,
                           scan_.metric_.value()};
  }
};

// Map a `preload` string to a `Residency`. An unknown value (should not
// happen -- validated at build time and by the runtime-override parser) falls
// back to `None`.
VectorIndex::Residency VectorIndex::residencyFromString(const std::string& s) {
  if (s == "advise") return Residency::Advise;
  if (s == "lock") return Residency::Lock;
  if (s == "aligned") return Residency::AlignedCopy;
  return Residency::None;
}

// Forward declaration (defined below `open`): apply a RAM-residency strategy
// to one matrix layer. (A template so that it can take the private
// `VectorIndex::Impl::MatrixLayer` without naming it.)
template <typename LayerT>
void makeLayerResident(LayerT& layer, VectorIndex::Residency residency,
                       size_t numRows, const std::string& indexName,
                       const char* layerLabel);

// ____________________________________________________________________________
VectorIndex::VectorIndex() : impl_{std::make_unique<Impl>()} {}
VectorIndex::~VectorIndex() = default;
VectorIndex::VectorIndex(VectorIndex&&) noexcept = default;
VectorIndex& VectorIndex::operator=(VectorIndex&&) noexcept = default;

// ____________________________________________________________________________
void VectorIndex::open(const std::string& basename, const std::string& name,
                       Residency residency, Residency rerankResidency) {
  auto& impl = *impl_;

  // 1. Metadata.
  std::ifstream metaIn{vectorMetaFile(basename, name)};
  if (!metaIn.is_open()) {
    AD_THROW("Could not open the metadata file of vector index \"" + name +
             "\": " + vectorMetaFile(basename, name));
  }
  nlohmann::json j;
  try {
    metaIn >> j;
    impl.meta_ = j.get<VectorIndexMetadata>();
  } catch (const std::exception& e) {
    AD_THROW("The metadata file of vector index \"" + name + "\" (" +
             vectorMetaFile(basename, name) + ") is malformed: " + e.what());
  }
  if (!isSupportedVectorIndexVersion(impl.meta_.version_)) {
    AD_THROW("Vector index \"" + name + "\" has on-disk version " +
             std::to_string(impl.meta_.version_) +
             ", but this binary expects version " +
             std::to_string(VECTOR_INDEX_VERSION) +
             " (or the still-supported version 4). Please rebuild the vector "
             "index.");
  }

  // 2. Flat store + entity mappings (memory-mapped).
  auto openMmap = [&](auto& view, const std::string& path,
                      ad_utility::AccessPattern pattern) {
    try {
      view.open(path, pattern);
    } catch (const std::exception& e) {
      AD_THROW("Could not open the file " + path + " of vector index \"" +
               name + "\": " + e.what());
    }
  };
  openMmap(impl.keys_, vectorKeysFile(basename, name),
           ad_utility::AccessPattern::Random);
  openMmap(impl.rowmap_, vectorRowmapFile(basename, name),
           ad_utility::AccessPattern::Random);
  openMmap(impl.scan_.data_, vectorDataFile(basename, name),
           ad_utility::AccessPattern::Random);
  auto complainInterrupted = [&](const std::string& what) {
    AD_THROW("Vector index \"" + name + "\": " + what +
             " does not match the metadata. The index files are likely from "
             "an interrupted build; please rebuild the vector index.");
  };
  if (impl.keys_.size() != impl.meta_.numVectors_) {
    complainInterrupted("the number of keys on disk");
  }
  if (impl.meta_.numTombstones_ > impl.meta_.numVectors_ ||
      impl.rowmap_.size() != impl.numLive()) {
    complainInterrupted("the entity mapping on disk");
  }
  if (impl.meta_.config_.dimensions_ == 0) {
    complainInterrupted("the dimension");
  }
  impl.scan_.dim_ = impl.dim();
  // Raw row byte length via `rowBytesFor`: `dim * bytesPerScalar`, except for
  // the sign-packed `binary` scalar whose rows are `(dim + 7) / 8` bytes.
  impl.scan_.rowBytes_ = rowBytesFor(impl.meta_.config_.scalar_, impl.dim());
  // Row stride: explicit for a v5 index, derived (== raw row byte length) for a
  // v4 index (`rowStrideBytes_ == 0`). It must be at least the raw row length
  // (the metric reads `rowBytes_` per row), or the mapped reads could go out
  // of bounds; the padded tail beyond it is never read.
  impl.scan_.stride_ = impl.meta_.rowStrideBytes_ != 0
                           ? impl.meta_.rowStrideBytes_
                           : impl.scan_.rowBytes_;
  if (impl.scan_.stride_ < impl.scan_.rowBytes_) {
    complainInterrupted("the row stride");
  }
  if (impl.scan_.data_.size() != impl.meta_.numVectors_ * impl.scan_.stride_) {
    complainInterrupted("the data size on disk");
  }
  // Validate the VALUES in the mapping (not just the counts): `.rowmap` must be
  // sorted by id and every `row_` must be in range, so that the unchecked
  // memory-mapped reads on the query paths can never go out of bounds on a
  // corrupt or truncated index file. One O(numLive) scan at load.
  for (size_t i = 0; i < impl.rowmap_.size(); ++i) {
    const IdRowPair& pair = impl.rowmap_[i];
    if (pair.row_ >= impl.meta_.numVectors_ || pair.idBits_ == TOMBSTONE_KEY ||
        (i > 0 && impl.rowmap_[i - 1].idBits_ >= pair.idBits_)) {
      complainInterrupted("the entity mapping on disk");
    }
  }

  // 3. The shared metric (over the storage scalar) and the f32 casts. The
  //    SCAN layer of a `binary` index is compared by HAMMING distance over the
  //    sign bits (`toUsearchScanMetric`), not the index's cosine metric --
  //    the fine rerank layer below always carries the index metric.
  impl.scan_.metric_.emplace(impl.meta_.config_.dimensions_,
                             toUsearchScanMetric(impl.meta_.config_.scalar_,
                                                 impl.meta_.config_.metric_),
                             toUsearchScalar(impl.meta_.config_.scalar_));
  impl.scan_.casts_ =
      uu::casts_punned_t::make(toUsearchScalar(impl.meta_.config_.scalar_));

  // 3b. The optional fine RERANK layer of a two-layer index: the same rows in
  //     the same order at the rerank precision, always with the natural
  //     (unpadded) stride. Same metric kind, its own scalar/casts. (Reset
  //     first so re-opening a single-layer index over a previously two-layer
  //     one never keeps a stale layer.)
  impl.rerank_.reset();
  if (impl.meta_.config_.rerankScalar_.has_value()) {
    const VectorScalar rerankScalar = impl.meta_.config_.rerankScalar_.value();
    auto& rerank = impl.rerank_.emplace();
    openMmap(rerank.data_, vectorRerankDataFile(basename, name),
             ad_utility::AccessPattern::Random);
    rerank.dim_ = impl.dim();
    rerank.rowBytes_ = rowBytesFor(rerankScalar, impl.dim());
    rerank.stride_ = rerank.rowBytes_;
    if (rerank.data_.size() != impl.meta_.numVectors_ * rerank.stride_) {
      complainInterrupted("the rerank data size on disk");
    }
    rerank.metric_.emplace(impl.meta_.config_.dimensions_,
                           toUsearchMetric(impl.meta_.config_.metric_),
                           toUsearchScalar(rerankScalar));
    rerank.casts_ = uu::casts_punned_t::make(toUsearchScalar(rerankScalar));
  }

  // 3c. The optional csls hubness sidecar `.csls`: one f32 r(d) per row,
  //     aligned to the flat store. (Reset first so re-opening a non-csls
  //     index over a csls one never keeps a stale mapping.)
  impl.cslsR_ = ad_utility::MmapVectorView<float>{};
  if (impl.meta_.config_.csls_) {
    openMmap(impl.cslsR_, vectorCslsFile(basename, name),
             ad_utility::AccessPattern::Random);
    if (impl.cslsR_.size() != impl.meta_.numVectors_) {
      complainInterrupted("the csls sidecar size on disk");
    }
  }

  // 4. The optional HNSW graph, memory-mapped read-only via usearch `view`.
  //    The graph is keyed by row; vectors come from the flat store above.
  if (impl.meta_.hasHnsw_) {
    const std::string hnswPath = vectorHnswFile(basename, name);
    impl.graph_.emplace(
        uu::index_config_t{impl.meta_.config_.hnswConnectivity_,
                           impl.meta_.config_.hnswConnectivity_ * 2});
    auto result = impl.graph_->view(uu::memory_mapped_file_t{hnswPath.c_str()});
    if (!result) {
      AD_THROW("Could not memory-map the HNSW file for vector index \"" + name +
               "\": " + result.error.what());
    }
    if (impl.graph_->size() != impl.meta_.numVectors_) {
      complainInterrupted("the HNSW graph on disk");
    }
    // Validate the graph member keys: they are row indices and are used
    // (unchecked, inside usearch's traversal) to read the flat store, so a
    // corrupt `.hnsw` with an out-of-range key must be rejected here rather
    // than cause an out-of-bounds read at query time. O(numVectors), once.
    for (auto it = impl.graph_->cbegin(); it != impl.graph_->cend(); ++it) {
      if (static_cast<uint64_t>(uu::get_key(*it)) >= impl.meta_.numVectors_) {
        complainInterrupted("the HNSW graph on disk");
      }
    }
    // Reserve one search context per pooled slot. QLever's query-thread count
    // (`-j`) may exceed the hardware concurrency; searches beyond the pool
    // size wait briefly for a free slot instead of failing.
    size_t numSlots = std::max<size_t>(
        2 * std::max(1u, std::thread::hardware_concurrency()), 16);
    uu::index_limits_t limits;
    limits.members = impl.graph_->size();
    limits.threads_add = 0;
    limits.threads_search = numSlots;
    if (!impl.graph_->try_reserve(limits)) {
      AD_THROW("Could not allocate the search contexts for vector index \"" +
               name + "\".");
    }
    impl.searchSlots_ = std::make_unique<SearchSlotPool>(numSlots);
  }

  // 5. Optionally make the flat store(s) resident / SIMD-aligned in RAM.
  //    Purely a paging/throughput optimisation applied after a successful
  //    open. The `residency`/`rerankResidency` arguments (the "preload" /
  //    "preloadRerank" serving settings the load hook threads in from
  //    `QLEVER_VECTOR_SEARCH_ENDPOINTS`, default `None` = mmap-only) are
  //    authoritative and applied PER LAYER -- residency is never persisted,
  //    so e.g. the small i8 scan matrix can be mlocked while the bf16 rerank
  //    matrix stays demand-paged.
  makeResident(residency);
  if (impl.rerank_.has_value()) {
    makeLayerResident(impl.rerank_.value(), rerankResidency,
                      impl.meta_.numVectors_, impl.meta_.config_.name_,
                      "rerank");
  }
}

// ____________________________________________________________________________
// Apply a RAM-residency strategy to one matrix layer (`layerLabel` only names
// the layer in the log lines). See `VectorIndex::makeResident`.
template <typename LayerT>
void makeLayerResident(LayerT& layer, VectorIndex::Residency residency,
                       size_t numRows, const std::string& indexName,
                       const char* layerLabel) {
  using Residency = VectorIndex::Residency;
  const size_t matrixBytes = layer.data_.size();
  if (residency == Residency::None || matrixBytes == 0) {
    return;
  }
  // Fits-in-RAM gate: never drive the machine into OOM/thrash. If we cannot
  // determine the physical memory, be conservative and skip. `AlignedCopy`
  // additionally holds a SECOND full copy, so require it to fit within ~45% of
  // RAM (matrix + its aligned copy <= ~90% of RAM); the others only prefault /
  // pin the single mapped copy, gated at ~90%. Gated per layer with that
  // layer's byte size.
  const uint64_t ram = totalPhysicalMemoryBytes();
  const uint64_t budget =
      residency == Residency::AlignedCopy ? (ram / 100) * 45 : (ram / 100) * 90;
  if (ram == 0 || matrixBytes > budget) {
    AD_LOG_WARN << "Vector index \"" << indexName << "\": the "
                << (matrixBytes >> 20) << " MiB " << layerLabel
                << " store does not comfortably fit in physical "
                   "memory; skipping the requested RAM preload."
                << std::endl;
    return;
  }
  layer.residency_ = residency;

  switch (residency) {
    case Residency::None:
      break;
    case Residency::Advise:
      layer.data_.prefault();
      break;
    case Residency::Lock:
      layer.data_.prefault();
      if (!layer.data_.lockInMemory()) {
        AD_LOG_WARN << "Vector index \"" << indexName << "\": could not mlock "
                    << "the " << layerLabel
                    << " store (RLIMIT_MEMLOCK?); "
                       "continuing with an unlocked mapping."
                    << std::endl;
      }
      break;
    case Residency::AlignedCopy: {
      // Copy the matrix into a 64-byte aligned buffer with a padded stride so
      // every row starts on a SIMD boundary (also fixes alignment for legacy
      // v4 files whose on-disk stride is unpadded).
      const size_t rowBytes = layer.rowBytes_;
      const size_t stride64 = alignUp(rowBytes);
      const size_t n = numRows;
      void* buf = nullptr;
      if (posix_memalign(&buf, SIMD_ALIGNMENT, n * stride64) != 0 ||
          buf == nullptr) {
        AD_LOG_WARN << "Vector index \"" << indexName
                    << "\": could not allocate the aligned RAM copy of the "
                    << layerLabel << " store; falling back to a prefault."
                    << std::endl;
        layer.data_.prefault();
        return;
      }
      std::unique_ptr<char, AlignedFree> owned{static_cast<char*>(buf)};
      // Zero the pad tails, then copy each row's `rowBytes` payload.
      std::memset(owned.get(), 0, n * stride64);
      for (size_t i = 0; i < n; ++i) {
        std::memcpy(owned.get() + i * stride64, layer.rowPtr(i), rowBytes);
      }
#if defined(MADV_HUGEPAGE)
      madvise(owned.get(), n * stride64, MADV_HUGEPAGE);
#endif
      // Repoint the read path at the aligned copy (rowPtr()/graphMetric() read
      // `base()` + `stride_`).
      layer.alignedBuf_ = std::move(owned);
      layer.stride_ = stride64;
      break;
    }
  }
}

// ____________________________________________________________________________
void VectorIndex::makeResident(Residency residency) {
  makeLayerResident(impl_->scan_, residency, impl_->meta_.numVectors_,
                    impl_->meta_.config_.name_, "flat");
}

// ____________________________________________________________________________
VectorIndex::Residency VectorIndex::residency() const {
  return impl_->scan_.residency_;
}

// ____________________________________________________________________________
VectorIndex::Residency VectorIndex::rerankResidency() const {
  return impl_->rerank_.has_value() ? impl_->rerank_->residency_
                                    : Residency::None;
}

// ____________________________________________________________________________
const VectorIndexMetadata& VectorIndex::metadata() const {
  return impl_->meta_;
}
// ____________________________________________________________________________
void VectorIndex::setEmbeddingEndpoint(std::optional<std::string> url,
                                       std::optional<std::string> model) {
  if (url.has_value()) {
    impl_->meta_.config_.embeddingUrl_ = std::move(url.value());
  }
  if (model.has_value()) {
    impl_->meta_.config_.embeddingModel_ = std::move(model.value());
  }
}
size_t VectorIndex::dimensions() const { return impl_->dim(); }
size_t VectorIndex::numVectors() const { return impl_->meta_.numVectors_; }
size_t VectorIndex::numLiveVectors() const { return impl_->numLive(); }
VectorMetric VectorIndex::metric() const {
  return impl_->meta_.config_.metric_;
}
bool VectorIndex::hasHnsw() const { return impl_->meta_.hasHnsw_; }
bool VectorIndex::hasRerankLayer() const { return impl_->rerank_.has_value(); }

// ____________________________________________________________________________
bool VectorIndex::hasCsls() const { return impl_->meta_.config_.csls_; }
size_t VectorIndex::cslsNeighbors() const {
  return impl_->meta_.config_.cslsNeighbors_;
}
float VectorIndex::cslsRForRow(size_t row) const {
  AD_CONTRACT_CHECK(hasCsls() && row < impl_->meta_.numVectors_);
  return impl_->cslsR_[row];
}
std::optional<float> VectorIndex::cslsRForEntity(Id entity) const {
  AD_CONTRACT_CHECK(hasCsls());
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  return impl_->cslsR_[row.value()];
}

// ____________________________________________________________________________
bool VectorIndex::hasVector(Id entity) const {
  return impl_->rowOf(entity).has_value();
}

// ____________________________________________________________________________
void VectorIndex::memberEntities(ql::span<Id> out) const {
  const auto& rowmap = impl_->rowmap_;
  AD_CONTRACT_CHECK(out.size() == rowmap.size());
  // The rowmap is stored strictly ascending by `idBits_` (validated at open),
  // so this single pass emits the members in ascending `ValueId` order.
  for (size_t i = 0; i < rowmap.size(); ++i) {
    out[i] = Id::fromBits(rowmap[i].idBits_);
  }
}

// ____________________________________________________________________________
std::optional<std::vector<float>> VectorIndex::getVector(Id entity) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  // Decode from the FINE layer (the rerank matrix when present): it is the
  // higher-precision representation of the same input vector.
  const auto& layer = impl.fine();
  std::vector<float> out(impl.dim());
  auto toF32 = layer.casts_.to.f32;
  if (toF32 == nullptr || !toF32(layer.rowPtr(row.value()), impl.dim(),
                                 reinterpret_cast<char*>(out.data()))) {
    // Same representation (f32): copy the raw bytes.
    std::memcpy(out.data(), layer.rowPtr(row.value()), layer.rowBytes_);
  }
  return out;
}

// ____________________________________________________________________________
Id distanceToValueId(float distance) {
  return std::isnan(distance)
             ? Id::makeUndefined()
             : Id::makeFromDouble(static_cast<double>(distance));
}

// ____________________________________________________________________________
// The `DistanceComputer` (and the one-shot `distance` overloads below) always
// read the FINE layer -- `impl_->fine()` is the rerank matrix when present --
// so `vec:distance` is the exact baseline of a two-layer index.
float VectorIndex::DistanceComputer::operator()(Id entity) const {
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return impl_->fine().distanceToRow(queryBytes_.data(), row.value());
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::atRow(size_t row) const {
  return impl_->fine().distanceToRow(queryBytes_.data(), row);
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::operator()(
    ql::span<const float> vector) const {
  if (vector.size() != impl_->dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = impl_->fine().encodeQuery(vector, buffer);
  return impl_->fine().distanceBetweenBytes(queryBytes_.data(), encoded);
}

// ____________________________________________________________________________
float VectorIndex::distance(Id a, Id b) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto rowA = impl.rowOf(a);
  auto rowB = impl.rowOf(b);
  if (!rowA.has_value() || !rowB.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return layer.distanceBetweenBytes(layer.rowPtr(rowA.value()),
                                    layer.rowPtr(rowB.value()));
}

// ____________________________________________________________________________
float VectorIndex::distance(Id entity, ql::span<const float> vector) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value() || vector.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = layer.encodeQuery(vector, buffer);
  return layer.distanceToRow(encoded, row.value());
}

// ____________________________________________________________________________
float VectorIndex::distance(ql::span<const float> a,
                            ql::span<const float> b) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  if (a.size() != impl.dim() || b.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> bufferA;
  std::vector<char> bufferB;
  const char* encodedA = layer.encodeQuery(a, bufferA);
  const char* encodedB = layer.encodeQuery(b, bufferB);
  return layer.distanceBetweenBytes(encodedA, encodedB);
}

// ____________________________________________________________________________
VectorIndex::DistanceComputer VectorIndex::makeDistanceComputer(
    ql::span<const float> query) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  // Encode the query into the FINE layer's storage scalar and OWN the
  // resulting bytes (`encodeQuery` may return a pointer into the raw f32
  // input when no conversion is needed, so always copy `rowBytes_` out of it).
  std::vector<char> buffer;
  const char* encoded = layer.encodeQuery(query, buffer);
  std::vector<char> owned(encoded, encoded + layer.rowBytes_);
  return DistanceComputer{&impl, std::move(owned)};
}

// ____________________________________________________________________________
std::optional<VectorIndex::DistanceComputer>
VectorIndex::makeDistanceComputerByEntity(Id entity) const {
  const auto& impl = *impl_;
  const auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  const char* stored = layer.rowPtr(row.value());
  std::vector<char> owned(stored, stored + layer.rowBytes_);
  return DistanceComputer{&impl, std::move(owned)};
}

namespace {
// Rows per contiguous chunk of the sorted merge-walk: each chunk pays ONE
// `lower_bound` into the rowmap and then walks a forward cursor, so the whole
// scan does ~one binary search per chunk instead of one per row. Also the
// cancellation-poll granularity. Coarse on purpose (the per-row cost is one
// pointer compare plus at most one SIMD kernel call).
constexpr size_t GATHER_CHUNK = 1024;

// Below this many rows the fork/join overhead of the parallel walk is not
// worth it, so a small block stays serial (mirrors the per-row path's
// threshold in `VectorDistanceExpression.cpp`).
constexpr size_t GATHER_PARALLEL_THRESHOLD = 2048;
}  // namespace

// ____________________________________________________________________________
void VectorIndex::gatherSortedDistances(ql::span<const Id> ascendingEntities,
                                        const DistanceComputer& computer,
                                        absl::FunctionRef<Id(size_t)> onMiss,
                                        absl::FunctionRef<bool()> isCancelled,
                                        int numThreads,
                                        ql::span<Id> out) const {
  const auto& impl = *impl_;
  const size_t n = ascendingEntities.size();
  AD_CORRECTNESS_CHECK(out.size() == n);
  if (n == 0) {
    return;
  }
  // The id-sorted rowmap as a contiguous, random-access span so each chunk can
  // `lower_bound` its first id and then advance a local cursor.
  const IdRowPair* rmBegin = impl.rowmap_.data();
  const IdRowPair* rmEnd = rmBegin + impl.rowmap_.size();

  // Process one contiguous chunk [lo, hi) of the ascending entity column. ONE
  // binary search locates the chunk's first id in the rowmap; a LOCAL cursor
  // `j` then walks forward as the (ascending) ids advance, so the matched row
  // is exactly the one `rowOf` would return (same rowmap, same comparator).
  // `j` is per-chunk scratch and every `out[i]` write is disjoint, so chunks
  // are independent and safe to run concurrently.
  auto processChunk = [&](size_t lo, size_t hi) {
    const IdRowPair* j =
        std::lower_bound(rmBegin, rmEnd, ascendingEntities[lo].getBits(),
                         [](const IdRowPair& pair, uint64_t bits) {
                           return pair.idBits_ < bits;
                         });
    for (size_t i = lo; i < hi; ++i) {
      // Consecutive duplicate ids resolve to the same row/distance -- reuse the
      // previous result instead of recomputing the SIMD distance.
      if (i > lo && ascendingEntities[i] == ascendingEntities[i - 1]) {
        out[i] = out[i - 1];
        continue;
      }
      uint64_t bits = ascendingEntities[i].getBits();
      while (j != rmEnd && j->idBits_ < bits) {
        ++j;
      }
      if (j != rmEnd && j->idBits_ == bits) {
        out[i] =
            distanceToValueId(computer.atRow(static_cast<size_t>(j->row_)));
      } else {
        // Not a live member: defer to the caller's per-row handling (a
        // non-member id, UNDEF, or a per-row float-list literal), keeping the
        // fast path bit-identical to the per-row path for these rows too.
        out[i] = onMiss(i);
      }
    }
  };

  const size_t numChunks = (n + GATHER_CHUNK - 1) / GATHER_CHUNK;
#ifdef _OPENMP
  if (numThreads > 1 && n >= GATHER_PARALLEL_THRESHOLD) {
#pragma omp parallel for schedule(dynamic) num_threads(numThreads)
    for (size_t c = 0; c < numChunks; ++c) {
      // Non-throwing per-chunk cancellation poll: an exception must never
      // unwind out of the OpenMP region, so a cancelled scan just leaves the
      // remaining chunks' caller-prefilled `out[i]` and the caller raises the
      // real cancellation afterwards.
      if (isCancelled()) {
        continue;
      }
      size_t lo = c * GATHER_CHUNK;
      processChunk(lo, std::min(n, lo + GATHER_CHUNK));
    }
    return;
  }
#else
  (void)numThreads;
#endif
  for (size_t c = 0; c < numChunks; ++c) {
    if (isCancelled()) {
      return;
    }
    size_t lo = c * GATHER_CHUNK;
    processChunk(lo, std::min(n, lo + GATHER_CHUNK));
  }
}

// Declared in `VectorIndex.h`. `/proc/cpuinfo` lists one block per LOGICAL
// cpu, blocks separated by blank lines; hyperthread siblings share their
// `(physical id, core id)` pair, so the number of unique pairs is the
// physical-core count.
unsigned physicalCoreCount() {
  static const unsigned count = [] {
    ad_utility::HashSet<uint64_t> cores;
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
      // The integer after the ':' if `line` starts with `key`, else `nullopt`.
      auto valueForKey = [](const std::string& line,
                            std::string_view key) -> std::optional<int64_t> {
        if (!ql::starts_with(line, key)) {
          return std::nullopt;
        }
        size_t colon = line.find(':', key.size());
        if (colon == std::string::npos) {
          return std::nullopt;
        }
        return std::strtoll(line.c_str() + colon + 1, nullptr, 10);
      };
      int64_t physicalId = 0;  // stays 0 on kernels that omit the field
      std::optional<int64_t> coreId;
      auto flushBlock = [&]() {
        if (coreId.has_value()) {
          cores.insert((static_cast<uint64_t>(physicalId) << 32) |
                       static_cast<uint32_t>(coreId.value()));
        }
        physicalId = 0;
        coreId = std::nullopt;
      };
      std::string line;
      while (std::getline(cpuinfo, line)) {
        if (line.empty()) {
          flushBlock();
        } else if (auto v = valueForKey(line, "physical id")) {
          physicalId = v.value();
        } else if (auto v = valueForKey(line, "core id")) {
          coreId = v;
        }
      }
      flushBlock();
    }
    if (!cores.empty()) {
      return static_cast<unsigned>(cores.size());
    }
    // No `/proc/cpuinfo`, or it has no core-topology fields (some
    // containers/ARM): fall back to the logical cpu count, which may include
    // hyperthreads -- acceptable.
    return std::max(1u, std::thread::hardware_concurrency());
  }();
  return count;
}

// Declared in `VectorIndex.h`.
int vectorSearchThreadCap() {
  int physical = static_cast<int>(physicalCoreCount());
#ifdef _OPENMP
  return std::max(1, std::min(omp_get_max_threads(), physical));
#else
  return std::max(1, physical);
#endif
}

namespace {
#ifdef _OPENMP
// Parallelize the exact top-k distance sweep once it crosses this many
// candidates. Below it (notably the small two-layer RERANK pass over a few
// hundred pruned candidates) the scan stays serial -- the OpenMP fan-out would
// cost more than it saves. Mirrors the `vec:distance` expression's thresholds.
constexpr size_t VEC_SEARCH_PARALLEL_THRESHOLD = 2048;
constexpr size_t VEC_SEARCH_PARALLEL_CHUNK = 1024;
#endif

// A bounded max-heap that keeps the `k` smallest (best) distances seen.
class TopK {
 public:
  explicit TopK(size_t k) : k_{k} {}
  void offer(float distance, uint64_t entity) {
    if (heap_.size() < k_) {
      heap_.emplace(distance, entity);
    } else if (distance < heap_.top().first) {
      heap_.pop();
      heap_.emplace(distance, entity);
    }
  }
  // Extract ascending by distance.
  std::vector<ScoredEntity> sorted() {
    std::vector<ScoredEntity> out;
    out.reserve(heap_.size());
    while (!heap_.empty()) {
      const auto& [dist, id] = heap_.top();
      out.push_back(ScoredEntity{Id::fromBits(id), dist});
      heap_.pop();
    }
    ql::ranges::reverse(out);
    return out;
  }

  // Drain `other`'s entries into this heap -- used to fold a thread-local
  // partial top-k into the global one after a parallel scan.
  void merge(TopK& other) {
    while (!other.heap_.empty()) {
      const auto& [dist, id] = other.heap_.top();
      offer(dist, id);
      other.heap_.pop();
    }
  }

 private:
  size_t k_;
  std::priority_queue<std::pair<float, uint64_t>> heap_;
};

// Score `n` items into `top` (a `k`-bounded heap): `rowAndId(i)` returns the
// `(row, id)` of item `i`, or `nullopt` to skip it. This is the GATHER scan
// for candidate-restricted searches, whose rows are scattered (the whole-index
// sweep has its own pointer-walk below). The distance computation -- the
// mmap-heavy SIMD work -- runs in parallel once `n` crosses
// `VEC_SEARCH_PARALLEL_THRESHOLD`, via per-thread heaps merged at the end,
// so a big candidate scan uses every core while the small rerank pass stays
// serial. For distinct distances the parallel result is identical to the
// serial one (the same `k` smallest distances).
template <typename LayerT, typename Fn>
void scanIntoTopK(TopK& top, [[maybe_unused]] size_t k, size_t n, LayerT& layer,
                  const char* queryBytes, std::optional<float> maxDistance,
                  Fn&& rowAndId, const CheckInterruptCallback& checkInterrupt) {
  auto scoreOne = [&](TopK& dst, size_t i) {
    std::optional<std::pair<size_t, uint64_t>> ri = rowAndId(i);
    if (!ri.has_value()) {
      return;
    }
    float dist = layer.distanceToRow(queryBytes, ri->first);
    if (!maxDistance.has_value() || dist <= maxDistance.value()) {
      dst.offer(dist, ri->second);
    }
  };
#ifdef _OPENMP
  const int numThreads = vectorSearchThreadCap();
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::vector<TopK> locals(static_cast<size_t>(numThreads), TopK{k});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      TopK& localTop = locals[static_cast<size_t>(omp_get_thread_num())];
      // schedule(static): each thread owns ONE contiguous ~n/numThreads block
      // and streams it start-to-end -- best prefetch locality for this
      // bandwidth-bound sweep, and no dynamic-scheduling overhead. The
      // iteration->thread assignment does not affect the result (the
      // per-thread heaps are merged below). `VEC_SEARCH_PARALLEL_CHUNK`
      // remains the interrupt-poll stride inside each block.
#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < n; ++i) {
        // An exception must never unwind out of the OpenMP region: poll the
        // (throwing) interrupt once per chunk, latch a flag on cancellation,
        // and re-raise once AFTER the region.
        if (cancelled.load(std::memory_order_relaxed)) {
          continue;
        }
        if (checkInterrupt && i % VEC_SEARCH_PARALLEL_CHUNK == 0) {
          try {
            checkInterrupt();
          } catch (...) {
            cancelled.store(true, std::memory_order_relaxed);
            continue;
          }
        }
        scoreOne(localTop, i);
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise the cancellation outside the parallel
                         // region
    }
    for (TopK& local : locals) {
      top.merge(local);
    }
    return;
  }
#endif
  size_t sinceCheck = 0;
  for (size_t i = 0; i < n; ++i) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    scoreOne(top, i);
  }
}

// The dedicated WHOLE-INDEX top-k sweep (the filtered search keeps the
// generic `scanIntoTopK` above, whose gather callback it genuinely needs).
// Visiting ALL rows 0..numVectors in physical order lets every worker carry a
// RUNNING row pointer (one `p += stride_` add per row) instead of the generic
// path's `base() + row * stride_` multiply, and read the keys column directly
// instead of a per-row callback. `numThreads > 1` runs MANUAL static
// partitioning: each worker owns one contiguous, evenly-split row range
// processed start-to-end, so each core streams its own region of the store --
// the prefetch-friendliest order for this memory-bandwidth-bound scan. Rows
// go into per-thread heaps merged at the end, so the thread assignment is
// irrelevant to the result: the same distances are offered as in the serial
// walk, and the top-k is bit-identical (for distinct distances).
template <typename ImplT, typename LayerT>
void scanWholeIndexIntoTopK(TopK& top, size_t k, ImplT& impl, LayerT& layer,
                            const char* queryBytes,
                            std::optional<float> maxDistance,
                            const CheckInterruptCallback& checkInterrupt,
                            [[maybe_unused]] int numThreads) {
  const size_t n = impl.meta_.numVectors_;
#ifdef _OPENMP
  if (numThreads > 1) {
    std::vector<TopK> locals(static_cast<size_t>(numThreads), TopK{k});
    std::atomic<bool> cancelled{false};
#pragma omp parallel num_threads(numThreads)
    {
      const size_t tid = static_cast<size_t>(omp_get_thread_num());
      // Partition by the ACTUAL team size (OpenMP may grant fewer threads
      // than requested); `tid < team <= numThreads` indexes `locals` safely.
      const size_t team = static_cast<size_t>(omp_get_num_threads());
      TopK& localTop = locals[tid];
      const size_t first = n * tid / team;
      const size_t last = n * (tid + 1) / team;
      const char* p = layer.base() + first * layer.stride_;
      for (size_t row = first; row < last; ++row, p += layer.stride_) {
        // An exception must never unwind out of the OpenMP region: poll the
        // (throwing) interrupt once per chunk, latch a flag on cancellation
        // (also noticing another worker's latch), and re-raise once AFTER the
        // region.
        if ((row - first) % VEC_SEARCH_PARALLEL_CHUNK == 0) {
          if (cancelled.load(std::memory_order_relaxed)) {
            break;
          }
          if (checkInterrupt) {
            try {
              checkInterrupt();
            } catch (...) {
              cancelled.store(true, std::memory_order_relaxed);
              break;
            }
          }
        }
        uint64_t id = impl.keys_[row];
        if (id != TOMBSTONE_KEY) {
          float dist = layer.distanceBetweenBytes(queryBytes, p);
          if (!maxDistance.has_value() || dist <= maxDistance.value()) {
            localTop.offer(dist, id);
          }
        }
      }
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise the cancellation outside the parallel
                         // region
    }
    for (TopK& local : locals) {
      top.merge(local);
    }
    return;
  }
#endif
  const char* p = layer.base();
  size_t sinceCheck = 0;
  for (size_t row = 0; row < n; ++row, p += layer.stride_) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    uint64_t id = impl.keys_[row];
    if (id != TOMBSTONE_KEY) {
      float dist = layer.distanceBetweenBytes(queryBytes, p);
      if (!maxDistance.has_value() || dist <= maxDistance.value()) {
        top.offer(dist, id);
      }
    }
  }
}
}  // namespace

// RAII: advise the flat store of `layer` for a SEQUENTIAL scan while an
// ordered scan runs, restoring RANDOM (the gather default) on exit. This is a
// purely advisory `madvise` read-ahead hint on the memory-mapped store; it
// never affects results. Callers skip it (pass `active = false`) when the
// store is a resident aligned RAM copy (no paging to advise). We call
// `advise()` (a stateless `madvise`) rather than `setAccessPattern()` on
// purpose: the search methods are logically const but share one `Impl` across
// concurrent query threads, and `setAccessPattern` would write the vector's
// non-atomic `_pattern` member (a data race). Concurrent `advise()` calls
// only issue overlapping advisory hints, which is harmless.
template <typename LayerT>
struct SeqScanHint {
  LayerT& layer_;
  bool active_;
  SeqScanHint(LayerT& layer, bool active) : layer_{layer}, active_{active} {
    if (active_) {
      layer_.data_.adviseAccessPattern(ad_utility::AccessPattern::Sequential);
    }
  }
  ~SeqScanHint() {
    if (active_) {
      layer_.data_.adviseAccessPattern(ad_utility::AccessPattern::Random);
    }
  }
};

// ____________________________________________________________________________
// The scalar-agnostic core of the exact search: `queryBytes` is already in the
// storage representation of `layer` -- the FINE layer for `searchExact*` (the
// exact baseline), the coarse SCAN layer for `searchExactCoarse*` (the
// SERVICE's candidate pass). (A template so that it can take the private
// `VectorIndex::Impl` without naming it.)
template <typename ImplT, typename LayerT>
std::vector<ScoredEntity> searchExactBytes(
    ImplT& impl, LayerT& layer, const char* queryBytes, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored = nullptr) {
  // `numScored` (optional out-param) reports how many vectors actually had a
  // distance computed -- the WHOLE live set for a whole-index search, or just
  // the candidates that are members for a restricted one. It is NOT the raw
  // candidate count: a candidate without a stored vector is skipped, not
  // scored.
  auto reportScored = [numScored](size_t n) {
    if (numScored != nullptr) {
      *numScored = n;
    }
  };
  // Clamp `k` (user-supplied, unbounded) to the live vector count AND a hard
  // maximum: it bounds the `TopK` heap and would otherwise be a remote OOM
  // lever.
  k = std::min({k, impl.numLive(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    reportScored(0);
    return {};
  }
  TopK top{k};

  if (!candidates.has_value()) {
    // Whole-index brute force: rows are visited in order, so it is sequential.
    // The distance sweep runs in parallel (per-thread heaps) above the
    // threshold, on the dedicated pointer-walk scan.
    int numThreads = 1;
#ifdef _OPENMP
    if (impl.meta_.numVectors_ >= VEC_SEARCH_PARALLEL_THRESHOLD) {
      numThreads = vectorSearchThreadCap();
    }
#endif
    AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
                << "\": whole-index scan (" << impl.numLive() << " vectors, "
                << numThreads << " threads)" << std::endl;
    SeqScanHint hint{layer, !layer.alignedBuf_};
    scanWholeIndexIntoTopK(top, k, impl, layer, queryBytes, maxDistance,
                           checkInterrupt, numThreads);
    reportScored(impl.numLive());
    return top.sorted();
  }

  AD_LOG_INFO << "Vector index \"" << impl.meta_.config_.name_
              << "\": filtered scan (" << candidates->size() << " candidates)"
              << std::endl;
  // Restricted search: merge-join the candidate id set against the id-sorted
  // `.rowmap`. This replaces both the old per-candidate `lower_bound` gather
  // (sparse sets) and the whole-index membership scan (dense sets) with a
  // single O(#candidates + #rows) pass whose row reads are sequential (see
  // `mergeJoinRowmap`). NOTE: an empty candidate set deliberately yields an
  // empty result -- the caller has already restricted the search space to
  // nothing.
  std::vector<uint64_t> candBits;
  candBits.reserve(candidates->size());
  for (Id c : candidates.value()) {
    candBits.push_back(c.getBits());
  }
  // The candidate column from an index scan already arrives ValueId-sorted; if
  // it does not (e.g. a hand-built set, or one deduplicated in hit order), sort
  // a local copy so the merge -- and its sequential `.data` access -- still
  // applies. (This also collapses duplicate candidate ids.)
  if (!std::is_sorted(candBits.begin(), candBits.end())) {
    std::sort(candBits.begin(), candBits.end());
  }
  // The merge emits rows in non-decreasing order on a genuinely id-sorted
  // store. Track it: if it is ever violated (a stale collation), latch the flag
  // so later gathers skip the misleading SEQUENTIAL hint.
  bool seqHint = !layer.alignedBuf_ &&
                 !impl.gatherNonMonotonic_.load(std::memory_order_relaxed);
  // Collect the matched (row, id) pairs first: the merge-join is cheap, while
  // the per-row SIMD distance below is the expensive part we parallelize. They
  // arrive in ascending row order on a monotonic store, so the sequential-scan
  // hint still applies while scoring.
  std::vector<std::pair<size_t, uint64_t>> matched;
  matched.reserve(candBits.size());
  bool monotonic = true;
  {
    SeqScanHint hint{layer, seqHint};
    mergeJoinRowmap(candBits.begin(), candBits.end(), impl.rowmap_.begin(),
                    impl.rowmap_.end(), [&](size_t row, uint64_t id) {
                      if (!matched.empty() && row < matched.back().first) {
                        monotonic = false;
                      }
                      matched.emplace_back(row, id);
                    });
  }
  if (seqHint && !monotonic) {
    impl.gatherNonMonotonic_.store(true, std::memory_order_relaxed);
  }
  {
    SeqScanHint hint{layer, seqHint && monotonic};
    scanIntoTopK(
        top, k, matched.size(), layer, queryBytes, maxDistance,
        [&](size_t i) -> std::optional<std::pair<size_t, uint64_t>> {
          return matched[i];
        },
        checkInterrupt);
  }
  // The members among the candidates -- the rows that actually got a distance.
  reportScored(matched.size());
  return top.sorted();
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExact(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  return searchExactBytes(impl, layer, queryBytes, k, candidates, maxDistance,
                          checkInterrupt, numScored);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactCoarse(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = impl.scan_.encodeQuery(query, buffer);
  return searchExactBytes(impl, impl.scan_, queryBytes, k, candidates,
                          maxDistance, checkInterrupt, numScored);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactByEntity(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchExactBytes(impl, layer, layer.rowPtr(row.value()), k, candidates,
                          maxDistance, checkInterrupt, numScored);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactCoarseByEntity(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchExactBytes(impl, impl.scan_, impl.scan_.rowPtr(row.value()), k,
                          candidates, maxDistance, checkInterrupt, numScored);
}

// ____________________________________________________________________________
// The scalar-agnostic core of the HNSW search (see `searchExactBytes`).
template <typename ImplT>
std::vector<ScoredEntity> searchHnswBytes(
    const ImplT& impl, const char* queryBytesIn, size_t k,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) {
  AD_CONTRACT_CHECK(impl.graph_.has_value(),
                    "searchHnsw called but this index has no HNSW structure.");
  // Clamp `k` (user-supplied, unbounded) to the live vector count AND a hard
  // maximum -- it drives the allocations and the search expansion below.
  k = std::min({k, impl.numLive(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    return {};
  }
  // Fetch `k + numTombstones` results: at most `numTombstones` of the nearest
  // can be tombstoned, so this provably captures the k nearest LIVE results in
  // a SINGLE search -- no unbounded retry loop. Cap the over-fetch (and hence
  // the allocation and the search expansion) at a bounded value: on a heavily
  // tombstoned index this may return fewer than k results, which is exactly
  // the documented "rebuild after many removals" case.
  constexpr size_t MAX_HNSW_FETCH = 1'000'000;
  size_t wanted = std::min<size_t>(
      {impl.meta_.numVectors_, k + impl.meta_.numTombstones_, MAX_HNSW_FETCH});
  if (checkInterrupt) {
    checkInterrupt();
  }
  auto slot = impl.searchSlots_->acquire(checkInterrupt);
  FlatStoreMetric metric = impl.graphMetric();
  const auto* queryBytes = reinterpret_cast<const uu::byte_t*>(queryBytesIn);

  uu::index_search_config_t config;
  config.thread = slot.id();
  config.expansion =
      std::max<size_t>(impl.meta_.config_.hnswExpansionSearch_, wanted);
  auto result = impl.graph_->search(queryBytes, wanted, metric, config);
  if (!result) {
    AD_THROW("HNSW search on vector index \"" + impl.meta_.config_.name_ +
             "\" failed: " + result.error.what());
  }
  std::vector<uint64_t> rows(wanted);
  std::vector<uu::distance_punned_t> dists(wanted);
  size_t count = result.dump_to(rows.data(), dists.data());
  std::vector<ScoredEntity> out;
  out.reserve(k);
  for (size_t i = 0; i < count && out.size() < k; ++i) {
    float dist = static_cast<float>(dists[i]);
    // `maxDistance` legitimately yields fewer than k results (there simply are
    // not k neighbours within the distance) -- that is the correct answer, not
    // a reason to fetch more.
    if (maxDistance.has_value() && dist > maxDistance.value()) {
      continue;
    }
    // Defense in depth: a graph member key is a row index; guard the unchecked
    // `keys_` access against a corrupt `.hnsw`.
    if (rows[i] >= impl.meta_.numVectors_) {
      continue;
    }
    uint64_t id = impl.keys_[rows[i]];
    if (id == TOMBSTONE_KEY) {
      continue;
    }
    out.push_back(ScoredEntity{Id::fromBits(id), dist});
  }
  return out;
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnsw(
    ql::span<const float> query, size_t k, std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  const auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  // The graph reads the SCAN layer, so the query is encoded into ITS scalar.
  std::vector<char> buffer;
  const char* queryBytes = impl.scan_.encodeQuery(query, buffer);
  return searchHnswBytes(impl, queryBytes, k, maxDistance, checkInterrupt);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnswByEntity(
    Id entity, size_t k, std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchHnswBytes(impl, impl.scan_.rowPtr(row.value()), k, maxDistance,
                         checkInterrupt);
}

namespace {
// A raw-vector query counts a candidate as the query's exact self-match (and
// excludes it from the r(q) neighbourhood) iff its cosine distance is below
// this. Identical bytes give a distance of exactly 0; the epsilon only covers
// the sqrt rounding of the norm terms. Distinct real embeddings are never
// this close, so at most the genuine self is excluded. (An entity query point
// excludes its own ROW instead -- exact, no epsilon.)
constexpr float CSLS_SELF_EPSILON = 1e-6f;
}  // namespace

// ____________________________________________________________________________
// The core of the CSLS-filtered search (see the header comment of
// `searchCsls`): a FULL cosine-distance sweep of the scoring set on the FINE
// layer, then the query-adaptive cut `2 * cos_sim - r(q) - r(d) >=
// threshold`, with `cos_sim = 1 - distance` (the usearch/NumKong angular
// convention -- the cosine metric is guaranteed by the caller). `selfRow`,
// when set, is the query entity's own row, excluded from the r(q)
// neighbourhood by identity; for a raw query vector the single nearest
// candidate is excluded iff its distance is < `CSLS_SELF_EPSILON`.
template <typename ImplT, typename LayerT>
std::vector<CslsScoredEntity> searchCslsBytes(
    ImplT& impl, LayerT& layer, const char* queryBytes,
    std::optional<size_t> selfRow, float threshold, size_t neighbors,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) {
  // 1. The scoring set as (row, id) pairs: all live rows, or -- exactly like
  //    the restricted `searchExact` -- the merge-join of the candidate ids
  //    against the id-sorted rowmap (which dedups and drops non-members).
  std::vector<std::pair<size_t, uint64_t>> matched;
  if (!candidates.has_value()) {
    matched.reserve(impl.numLive());
    for (size_t row = 0; row < impl.meta_.numVectors_; ++row) {
      uint64_t id = impl.keys_[row];
      if (id != TOMBSTONE_KEY) {
        matched.emplace_back(row, id);
      }
    }
  } else {
    std::vector<uint64_t> candBits;
    candBits.reserve(candidates->size());
    for (Id c : candidates.value()) {
      candBits.push_back(c.getBits());
    }
    if (!std::is_sorted(candBits.begin(), candBits.end())) {
      std::sort(candBits.begin(), candBits.end());
    }
    matched.reserve(candBits.size());
    mergeJoinRowmap(candBits.begin(), candBits.end(), impl.rowmap_.begin(),
                    impl.rowmap_.end(), [&](size_t row, uint64_t id) {
                      matched.emplace_back(row, id);
                    });
  }
  if (numScored != nullptr) {
    *numScored = matched.size();
  }
  if (matched.empty()) {
    return {};
  }

  // 2. The full fine-layer distance sweep (CSLS needs EVERY cosine, so there
  //    is no top-k shortcut). Rows come out of step 1 (near-)ascending, so
  //    the sequential read-ahead hint applies; the mmap-heavy SIMD work runs
  //    in parallel above the usual threshold.
  const size_t n = matched.size();
  std::vector<float> dists(n);
  SeqScanHint hint{layer, !layer.alignedBuf_};
#ifdef _OPENMP
  const int numThreads = vectorSearchThreadCap();
  if (n >= VEC_SEARCH_PARALLEL_THRESHOLD && numThreads > 1) {
    std::atomic<bool> cancelled{false};
#pragma omp parallel for schedule(dynamic, VEC_SEARCH_PARALLEL_CHUNK) \
    num_threads(numThreads)
    for (size_t i = 0; i < n; ++i) {
      // An exception must never unwind out of the OpenMP region: poll the
      // (throwing) interrupt once per chunk, latch, re-raise after.
      if (cancelled.load(std::memory_order_relaxed)) {
        continue;
      }
      if (checkInterrupt && i % VEC_SEARCH_PARALLEL_CHUNK == 0) {
        try {
          checkInterrupt();
        } catch (...) {
          cancelled.store(true, std::memory_order_relaxed);
          continue;
        }
      }
      dists[i] = layer.distanceToRow(queryBytes, matched[i].first);
    }
    if (cancelled.load(std::memory_order_relaxed) && checkInterrupt) {
      checkInterrupt();  // re-raise outside the parallel region
    }
  } else
#endif
  {
    size_t sinceCheck = 0;
    for (size_t i = 0; i < n; ++i) {
      if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        checkInterrupt();
      }
      dists[i] = layer.distanceToRow(queryBytes, matched[i].first);
    }
  }

  // 3. r(q): the mean cosine similarity of the query to its top-`neighbors`
  //    nearest SCORED candidates, the exact self-match excluded. Computed
  //    BEFORE any `maxDistance` filter (it describes the retrieval geometry).
  size_t excluded = n;  // index into `matched`/`dists`; `n` = none
  if (selfRow.has_value()) {
    for (size_t i = 0; i < n; ++i) {
      if (matched[i].first == selfRow.value()) {
        excluded = i;
        break;
      }
    }
  } else {
    size_t best = 0;
    for (size_t i = 1; i < n; ++i) {
      if (dists[i] < dists[best]) {
        best = i;
      }
    }
    if (dists[best] < CSLS_SELF_EPSILON) {
      excluded = best;
    }
  }
  CslsNeighborhood neighborhood{neighbors};
  for (size_t i = 0; i < n; ++i) {
    if (i != excluded) {
      neighborhood.offer(dists[i]);
    }
  }
  const float rq = neighborhood.meanCosSim();

  // 4. The cut: keep candidate d iff `2 * cos_sim(q, d) - r(q) - r(d) >=
  //    threshold` (and its cosine distance passes `maxDistance`, if set).
  //    ALL survivors are returned, ascending by cosine DISTANCE -- CSLS is
  //    the cut, the cosine distance stays the score; the caller applies any
  //    top-k cap.
  std::vector<CslsScoredEntity> out;
  for (size_t i = 0; i < n; ++i) {
    const float d = dists[i];
    if (maxDistance.has_value() && d > maxDistance.value()) {
      continue;
    }
    const float csls = static_cast<float>(
        2.0 * (1.0 - static_cast<double>(d)) - static_cast<double>(rq) -
        static_cast<double>(impl.cslsR_[matched[i].first]));
    if (csls >= threshold) {
      out.push_back(CslsScoredEntity{Id::fromBits(matched[i].second), d, csls});
    }
  }
  ql::ranges::sort(out,
                   [](const CslsScoredEntity& a, const CslsScoredEntity& b) {
                     return a.distance_ != b.distance_
                                ? a.distance_ < b.distance_
                                : a.entity_.getBits() < b.entity_.getBits();
                   });
  return out;
}

// ____________________________________________________________________________
std::vector<CslsScoredEntity> VectorIndex::searchCsls(
    ql::span<const float> query, float threshold, size_t neighbors,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the sweep can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.meta_.config_.csls_,
                    "searchCsls called on an index without csls data.");
  // Guaranteed by the build (`csls` is rejected for non-cosine metrics).
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  auto& layer = impl.fine();
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = layer.encodeQuery(query, buffer);
  return searchCslsBytes(impl, layer, queryBytes, std::nullopt, threshold,
                         neighbors, candidates, maxDistance, checkInterrupt,
                         numScored);
}

// ____________________________________________________________________________
std::vector<CslsScoredEntity> VectorIndex::searchCslsByEntity(
    Id entity, float threshold, size_t neighbors,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt, size_t* numScored) const {
  // Non-const so the sweep can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.meta_.config_.csls_,
                    "searchCslsByEntity called on an index without csls data.");
  AD_CORRECTNESS_CHECK(impl.meta_.config_.metric_ == VectorMetric::Cosine);
  auto& layer = impl.fine();
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    if (numScored != nullptr) {
      *numScored = 0;
    }
    return {};
  }
  return searchCslsBytes(impl, layer, layer.rowPtr(row.value()), row, threshold,
                         neighbors, candidates, maxDistance, checkInterrupt,
                         numScored);
}

}  // namespace qlever::vector
