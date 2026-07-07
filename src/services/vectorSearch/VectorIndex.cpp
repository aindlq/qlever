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
#include <queue>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

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
  VectorIndexMetadata meta_;
  ad_utility::MmapVectorView<uint64_t> keys_;  // row -> id (or TOMBSTONE_KEY)
  ad_utility::MmapVectorView<IdRowPair> rowmap_;  // id -> row, sorted by id
  // Row-major matrix in the configured storage scalar (f32/f16/i8).
  ad_utility::MmapVectorView<char> data_;
  std::optional<uu::metric_punned_t> metric_;  // shared by exact + HNSW
  uu::casts_punned_t casts_;                   // f32 <-> storage scalar
  std::optional<GraphIndex> graph_;            // present iff meta_.hasHnsw_
  std::unique_ptr<SearchSlotPool> searchSlots_;
  // Byte stride between consecutive rows in `.data` (>= `rowBytes()`). Derived
  // from the metadata at open; for a `Residency::AlignedCopy` it becomes the
  // padded stride of `alignedBuf_`.
  size_t stride_ = 0;
  // Optional 64-byte-aligned RAM copy of the whole matrix (`Residency`
  // `AlignedCopy`). When set, `base()` reads from it (with the padded
  // `stride_`) instead of the memory-mapped file.
  std::unique_ptr<char, AlignedFree> alignedBuf_;
  // The residency strategy `makeResident` last applied (best-effort; see
  // `VectorIndex::residency()`). Stays `None` when the request was skipped by
  // the fits-in-RAM gate.
  Residency residency_ = Residency::None;
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
  size_t rowBytes() const {
    return dim() * bytesPerScalar(meta_.config_.scalar_);
  }
  size_t numLive() const { return meta_.numVectors_ - meta_.numTombstones_; }

  // Base pointer of the active matrix: the aligned RAM copy if present, else
  // the memory-mapped file.
  const char* base() const {
    return alignedBuf_ ? alignedBuf_.get() : data_.data();
  }

  // Pointer to the start of row `i` in the active matrix (storage scalar
  // bytes). Uses the per-row stride, which may exceed `rowBytes()` when rows
  // are padded for SIMD alignment; the metric reads only `dim()` scalars.
  const char* rowPtr(size_t i) const { return base() + i * stride_; }

  // Encode an f32 query into the storage scalar. Returns a pointer to the
  // encoded bytes; `buffer` provides the storage when a conversion happens.
  const char* encodeQuery(ql::span<const float> query,
                          std::vector<char>& buffer) const {
    const char* raw = reinterpret_cast<const char*>(query.data());
    auto fromF32 = casts_.from.f32;
    if (fromF32 != nullptr) {
      buffer.resize(rowBytes());
      if (fromF32(raw, dim(), buffer.data())) {
        return buffer.data();
      }
    }
    return raw;
  }

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

  // Distance between two (encoded, `rowBytes()`-sized) vectors, using the
  // (punned) index metric so that exact and HNSW distances are identical.
  float distanceBetweenBytes(const char* a, const char* b) const {
    return static_cast<float>(
        metric_.value()(reinterpret_cast<const uu::byte_t*>(a),
                        reinterpret_cast<const uu::byte_t*>(b)));
  }

  // Distance between an (encoded) query and row `i`.
  float distanceToRow(const char* queryBytes, size_t i) const {
    return distanceBetweenBytes(queryBytes, rowPtr(i));
  }

  FlatStoreMetric graphMetric() const {
    return FlatStoreMetric{base(), stride_, meta_.numVectors_, metric_.value()};
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

// ____________________________________________________________________________
VectorIndex::VectorIndex() : impl_{std::make_unique<Impl>()} {}
VectorIndex::~VectorIndex() = default;
VectorIndex::VectorIndex(VectorIndex&&) noexcept = default;
VectorIndex& VectorIndex::operator=(VectorIndex&&) noexcept = default;

// ____________________________________________________________________________
void VectorIndex::open(const std::string& basename, const std::string& name,
                       Residency residency) {
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
  openMmap(impl.data_, vectorDataFile(basename, name),
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
  // Row stride: explicit for a v5 index, derived (== raw row byte length) for a
  // v4 index (`rowStrideBytes_ == 0`). It must be at least the raw row length
  // (the metric reads `rowBytes()` per row), or the mapped reads could go out
  // of bounds; the padded tail beyond it is never read.
  impl.stride_ = impl.meta_.rowStrideBytes_ != 0 ? impl.meta_.rowStrideBytes_
                                                 : impl.rowBytes();
  if (impl.stride_ < impl.rowBytes()) {
    complainInterrupted("the row stride");
  }
  if (impl.data_.size() != impl.meta_.numVectors_ * impl.stride_) {
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

  // 3. The shared metric (over the storage scalar) and the f32 casts.
  impl.metric_.emplace(impl.meta_.config_.dimensions_,
                       toUsearchMetric(impl.meta_.config_.metric_),
                       toUsearchScalar(impl.meta_.config_.scalar_));
  impl.casts_ =
      uu::casts_punned_t::make(toUsearchScalar(impl.meta_.config_.scalar_));

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

  // 5. Optionally make the flat store resident / SIMD-aligned in RAM. Purely a
  //    paging/throughput optimisation applied after a successful open. The
  //    `residency` argument (the "preload" serving setting the load hook
  //    threads in from `QLEVER_VECTOR_SEARCH_ENDPOINTS`, default `None` =
  //    mmap-only) is authoritative -- residency is never persisted.
  makeResident(residency);
}

// ____________________________________________________________________________
void VectorIndex::makeResident(Residency residency) {
  auto& impl = *impl_;
  const size_t matrixBytes = impl.data_.size();
  if (residency == Residency::None || matrixBytes == 0) {
    return;
  }
  // Fits-in-RAM gate: never drive the machine into OOM/thrash. If we cannot
  // determine the physical memory, be conservative and skip. `AlignedCopy`
  // additionally holds a SECOND full copy, so require it to fit within ~45% of
  // RAM (matrix + its aligned copy <= ~90% of RAM); the others only prefault /
  // pin the single mapped copy, gated at ~90%.
  const uint64_t ram = totalPhysicalMemoryBytes();
  const uint64_t budget =
      residency == Residency::AlignedCopy ? (ram / 100) * 45 : (ram / 100) * 90;
  if (ram == 0 || matrixBytes > budget) {
    AD_LOG_WARN << "Vector index \"" << impl.meta_.config_.name_ << "\": the "
                << (matrixBytes >> 20)
                << " MiB flat store does not comfortably fit in physical "
                   "memory; skipping the requested RAM preload."
                << std::endl;
    return;
  }
  impl.residency_ = residency;

  switch (residency) {
    case Residency::None:
      break;
    case Residency::Advise:
      impl.data_.prefault();
      break;
    case Residency::Lock:
      impl.data_.prefault();
      if (!impl.data_.lockInMemory()) {
        AD_LOG_WARN << "Vector index \"" << impl.meta_.config_.name_
                    << "\": could not mlock the flat store (RLIMIT_MEMLOCK?); "
                       "continuing with an unlocked mapping."
                    << std::endl;
      }
      break;
    case Residency::AlignedCopy: {
      // Copy the matrix into a 64-byte aligned buffer with a padded stride so
      // every row starts on a SIMD boundary (also fixes alignment for legacy
      // v4 files whose on-disk stride is unpadded).
      const size_t rowBytes = impl.rowBytes();
      const size_t stride64 = alignUp(rowBytes);
      const size_t n = impl.meta_.numVectors_;
      void* buf = nullptr;
      if (posix_memalign(&buf, SIMD_ALIGNMENT, n * stride64) != 0 ||
          buf == nullptr) {
        AD_LOG_WARN << "Vector index \"" << impl.meta_.config_.name_
                    << "\": could not allocate the aligned RAM copy; falling "
                       "back to a prefault."
                    << std::endl;
        impl.data_.prefault();
        return;
      }
      std::unique_ptr<char, AlignedFree> owned{static_cast<char*>(buf)};
      // Zero the pad tails, then copy each row's `rowBytes` payload.
      std::memset(owned.get(), 0, n * stride64);
      for (size_t i = 0; i < n; ++i) {
        std::memcpy(owned.get() + i * stride64, impl.rowPtr(i), rowBytes);
      }
#if defined(MADV_HUGEPAGE)
      madvise(owned.get(), n * stride64, MADV_HUGEPAGE);
#endif
      // Repoint the read path at the aligned copy (rowPtr()/graphMetric() read
      // `base()` + `stride_`).
      impl.alignedBuf_ = std::move(owned);
      impl.stride_ = stride64;
      break;
    }
  }
}

// ____________________________________________________________________________
VectorIndex::Residency VectorIndex::residency() const {
  return impl_->residency_;
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

// ____________________________________________________________________________
bool VectorIndex::hasVector(Id entity) const {
  return impl_->rowOf(entity).has_value();
}

// ____________________________________________________________________________
std::optional<std::vector<float>> VectorIndex::getVector(Id entity) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  std::vector<float> out(impl.dim());
  auto toF32 = impl.casts_.to.f32;
  if (toF32 == nullptr || !toF32(impl.rowPtr(row.value()), impl.dim(),
                                 reinterpret_cast<char*>(out.data()))) {
    // Same representation (f32): copy the raw bytes.
    std::memcpy(out.data(), impl.rowPtr(row.value()), impl.rowBytes());
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
float VectorIndex::DistanceComputer::operator()(Id entity) const {
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return impl_->distanceToRow(queryBytes_.data(), row.value());
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::atRow(size_t row) const {
  return impl_->distanceToRow(queryBytes_.data(), row);
}

// ____________________________________________________________________________
float VectorIndex::DistanceComputer::operator()(
    ql::span<const float> vector) const {
  if (vector.size() != impl_->dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = impl_->encodeQuery(vector, buffer);
  return impl_->distanceBetweenBytes(queryBytes_.data(), encoded);
}

// ____________________________________________________________________________
float VectorIndex::distance(Id a, Id b) const {
  const auto& impl = *impl_;
  auto rowA = impl.rowOf(a);
  auto rowB = impl.rowOf(b);
  if (!rowA.has_value() || !rowB.has_value()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  return impl.distanceBetweenBytes(impl.rowPtr(rowA.value()),
                                   impl.rowPtr(rowB.value()));
}

// ____________________________________________________________________________
float VectorIndex::distance(Id entity, ql::span<const float> vector) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value() || vector.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> buffer;
  const char* encoded = impl.encodeQuery(vector, buffer);
  return impl.distanceToRow(encoded, row.value());
}

// ____________________________________________________________________________
float VectorIndex::distance(ql::span<const float> a,
                            ql::span<const float> b) const {
  const auto& impl = *impl_;
  if (a.size() != impl.dim() || b.size() != impl.dim()) {
    return std::numeric_limits<float>::quiet_NaN();
  }
  std::vector<char> bufferA;
  std::vector<char> bufferB;
  const char* encodedA = impl.encodeQuery(a, bufferA);
  const char* encodedB = impl.encodeQuery(b, bufferB);
  return impl.distanceBetweenBytes(encodedA, encodedB);
}

// ____________________________________________________________________________
VectorIndex::DistanceComputer VectorIndex::makeDistanceComputer(
    ql::span<const float> query) const {
  const auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  // Encode the query into the storage scalar and OWN the resulting bytes
  // (`encodeQuery` may return a pointer into the raw f32 input when no
  // conversion is needed, so always copy `rowBytes` out of it).
  std::vector<char> buffer;
  const char* encoded = impl.encodeQuery(query, buffer);
  std::vector<char> owned(encoded, encoded + impl.rowBytes());
  return DistanceComputer{&impl, std::move(owned)};
}

// ____________________________________________________________________________
std::optional<VectorIndex::DistanceComputer>
VectorIndex::makeDistanceComputerByEntity(Id entity) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  const char* stored = impl.rowPtr(row.value());
  std::vector<char> owned(stored, stored + impl.rowBytes());
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

namespace {
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

 private:
  size_t k_;
  std::priority_queue<std::pair<float, uint64_t>> heap_;
};
}  // namespace

// ____________________________________________________________________________
// The scalar-agnostic core of the exact search: `queryBytes` is already in the
// storage representation. (A template so that it can take the private
// `VectorIndex::Impl` without naming it.)
template <typename ImplT>
std::vector<ScoredEntity> searchExactBytes(
    ImplT& impl, const char* queryBytes, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) {
  // Clamp `k` (user-supplied, unbounded) to the live vector count AND a hard
  // maximum: it bounds the `TopK` heap and would otherwise be a remote OOM
  // lever.
  k = std::min({k, impl.numLive(), MAX_SEARCH_RESULTS});
  if (k == 0) {
    return {};
  }
  TopK top{k};
  size_t sinceCheck = 0;
  auto consider = [&](size_t row, uint64_t id) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    float dist = impl.distanceToRow(queryBytes, row);
    if (!maxDistance.has_value() || dist <= maxDistance.value()) {
      top.offer(dist, id);
    }
  };

  // RAII: advise the flat store for a SEQUENTIAL scan while the ordered scan
  // below runs, restoring RANDOM (the gather default) on exit. This is a purely
  // advisory `madvise` read-ahead hint on the memory-mapped store; it never
  // affects results. Skipped when the store is a resident aligned RAM copy (no
  // paging to advise). We call `advise()` (a stateless `madvise`) rather than
  // `setAccessPattern()` on purpose: the search methods are logically const but
  // share one `Impl` across concurrent query threads, and `setAccessPattern`
  // would write the vector's non-atomic `_pattern` member (a data race).
  // Concurrent `advise()` calls only issue overlapping advisory hints, which is
  // harmless.
  struct SeqScanHint {
    ImplT& impl_;
    bool active_;
    SeqScanHint(ImplT& impl, bool active) : impl_{impl}, active_{active} {
      if (active_) {
        impl_.data_.adviseAccessPattern(ad_utility::AccessPattern::Sequential);
      }
    }
    ~SeqScanHint() {
      if (active_) {
        impl_.data_.adviseAccessPattern(ad_utility::AccessPattern::Random);
      }
    }
  };

  if (!candidates.has_value()) {
    // Whole-index brute force: rows are visited in order, so it is sequential.
    SeqScanHint hint{impl, !impl.alignedBuf_};
    for (size_t row = 0; row < impl.meta_.numVectors_; ++row) {
      uint64_t id = impl.keys_[row];
      if (id != TOMBSTONE_KEY) {
        consider(row, id);
      }
    }
    return top.sorted();
  }

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
  bool seqHint = !impl.alignedBuf_ &&
                 !impl.gatherNonMonotonic_.load(std::memory_order_relaxed);
  bool monotonic = true;
  size_t prevRow = 0;
  bool havePrev = false;
  {
    SeqScanHint hint{impl, seqHint};
    mergeJoinRowmap(candBits.begin(), candBits.end(), impl.rowmap_.begin(),
                    impl.rowmap_.end(), [&](size_t row, uint64_t id) {
                      if (havePrev && row < prevRow) {
                        monotonic = false;
                      }
                      prevRow = row;
                      havePrev = true;
                      consider(row, id);
                    });
  }
  if (seqHint && !monotonic) {
    impl.gatherNonMonotonic_.store(true, std::memory_order_relaxed);
  }
  return top.sorted();
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExact(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = impl.encodeQuery(query, buffer);
  return searchExactBytes(impl, queryBytes, k, candidates, maxDistance,
                          checkInterrupt);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExactByEntity(
    Id entity, size_t k, std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  // Non-const so the gather can toggle the flat store's (advisory) access-
  // pattern hint; all result-affecting state stays read-only.
  auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchExactBytes(impl, impl.rowPtr(row.value()), k, candidates,
                          maxDistance, checkInterrupt);
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
  std::vector<char> buffer;
  const char* queryBytes = impl.encodeQuery(query, buffer);
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
  return searchHnswBytes(impl, impl.rowPtr(row.value()), k, maxDistance,
                         checkInterrupt);
}

}  // namespace qlever::vector
