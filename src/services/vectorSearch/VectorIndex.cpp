// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndex.h"

#include <condition_variable>
#include <cstring>
#include <fstream>
#include <mutex>
#include <queue>
#include <thread>

#include "backports/algorithm.h"
#include "services/vectorSearch/UsearchGraph.h"
#include "util/Exception.h"
#include "util/HashSet.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
// How often the exact-search loops call the interrupt callback. Frequent enough
// for sub-second cancellation latency, rare enough to not show up in profiles.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

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

  Slot acquire() {
    std::unique_lock lock{mutex_};
    cv_.wait(lock, [this] { return !free_.empty(); });
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

  size_t dim() const { return meta_.config_.dimensions_; }
  size_t rowBytes() const {
    return dim() * bytesPerScalar(meta_.config_.scalar_);
  }
  size_t numLive() const { return meta_.numVectors_ - meta_.numTombstones_; }

  // Pointer to the start of row `i` in the flat store (storage scalar bytes).
  const char* rowPtr(size_t i) const { return data_.data() + i * rowBytes(); }

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

  // Distance between an (encoded) query and row `i`, using the (punned) index
  // metric so that exact and HNSW distances are identical.
  float distanceToRow(const char* queryBytes, size_t i) const {
    return static_cast<float>(
        metric_.value()(reinterpret_cast<const uu::byte_t*>(queryBytes),
                        reinterpret_cast<const uu::byte_t*>(rowPtr(i))));
  }

  FlatStoreMetric graphMetric() const {
    return FlatStoreMetric{data_.data(), rowBytes(), metric_.value()};
  }
};

// ____________________________________________________________________________
VectorIndex::VectorIndex() : impl_{std::make_unique<Impl>()} {}
VectorIndex::~VectorIndex() = default;
VectorIndex::VectorIndex(VectorIndex&&) noexcept = default;
VectorIndex& VectorIndex::operator=(VectorIndex&&) noexcept = default;

// ____________________________________________________________________________
void VectorIndex::open(const std::string& basename, const std::string& name) {
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
  if (impl.meta_.version_ != VECTOR_INDEX_VERSION) {
    AD_THROW("Vector index \"" + name + "\" has on-disk version " +
             std::to_string(impl.meta_.version_) +
             ", but this binary expects version " +
             std::to_string(VECTOR_INDEX_VERSION) +
             ". Please rebuild the vector index.");
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
  if (impl.data_.size() != impl.meta_.numVectors_ * impl.rowBytes()) {
    complainInterrupted("the data size on disk");
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
}

// ____________________________________________________________________________
const VectorIndexMetadata& VectorIndex::metadata() const {
  return impl_->meta_;
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
    const ImplT& impl, const char* queryBytes, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) {
  if (k == 0 || impl.numLive() == 0) {
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
  auto scanAll = [&](auto&& keepId) {
    for (size_t row = 0; row < impl.meta_.numVectors_; ++row) {
      uint64_t id = impl.keys_[row];
      if (id != TOMBSTONE_KEY && keepId(id)) {
        consider(row, id);
      }
    }
  };
  if (!candidates.has_value()) {
    scanAll([](uint64_t) { return true; });
  } else if (candidates->size() * 8 >= impl.numLive()) {
    // A large candidate set: a sequential scan with a membership filter beats
    // one random binary search + one random row access per candidate.
    // NOTE: an empty candidate set deliberately yields an empty result -- the
    // caller has already restricted the search space to nothing.
    ad_utility::HashSet<uint64_t> wanted;
    wanted.reserve(candidates->size());
    for (Id c : candidates.value()) {
      wanted.insert(c.getBits());
    }
    scanAll([&wanted](uint64_t id) { return wanted.contains(id); });
  } else {
    for (Id c : candidates.value()) {
      if (auto row = impl.rowOf(c); row.has_value()) {
        consider(row.value(), c.getBits());
      }
    }
  }
  return top.sorted();
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchExact(
    ql::span<const float> query, size_t k,
    std::optional<ql::span<const Id>> candidates,
    std::optional<float> maxDistance,
    const CheckInterruptCallback& checkInterrupt) const {
  const auto& impl = *impl_;
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
  const auto& impl = *impl_;
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
std::vector<ScoredEntity> searchHnswBytes(const ImplT& impl,
                                          const char* queryBytesIn, size_t k,
                                          std::optional<float> maxDistance) {
  AD_CONTRACT_CHECK(impl.graph_.has_value(),
                    "searchHnsw called but this index has no HNSW structure.");
  if (k == 0 || impl.numLive() == 0) {
    return {};
  }
  auto slot = impl.searchSlots_->acquire();
  FlatStoreMetric metric = impl.graphMetric();
  const auto* queryBytes = reinterpret_cast<const uu::byte_t*>(queryBytesIn);

  // Tombstoned rows are filtered AFTER the graph search; if too few live
  // results remain, retry with a doubled result count (bounded by the graph
  // size). Without tombstones a single pass always suffices.
  size_t wanted = k;
  std::vector<ScoredEntity> out;
  while (true) {
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
    out.clear();
    for (size_t i = 0; i < count && out.size() < k; ++i) {
      float dist = static_cast<float>(dists[i]);
      if (maxDistance.has_value() && dist > maxDistance.value()) {
        continue;
      }
      uint64_t id = impl.keys_[rows[i]];
      if (id == TOMBSTONE_KEY) {
        continue;
      }
      out.push_back(ScoredEntity{Id::fromBits(id), dist});
    }
    bool exhausted = wanted >= impl.meta_.numVectors_;
    if (out.size() >= k || impl.meta_.numTombstones_ == 0 || exhausted) {
      break;
    }
    wanted = std::min<size_t>(impl.meta_.numVectors_, wanted * 2);
  }
  return out;
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnsw(
    ql::span<const float> query, size_t k,
    std::optional<float> maxDistance) const {
  const auto& impl = *impl_;
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  std::vector<char> buffer;
  const char* queryBytes = impl.encodeQuery(query, buffer);
  return searchHnswBytes(impl, queryBytes, k, maxDistance);
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnswByEntity(
    Id entity, size_t k, std::optional<float> maxDistance) const {
  const auto& impl = *impl_;
  auto row = impl.rowOf(entity);
  if (!row.has_value()) {
    return {};
  }
  return searchHnswBytes(impl, impl.rowPtr(row.value()), k, maxDistance);
}

}  // namespace qlever::vector
