// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorIndex.h"

#include <fstream>
#include <queue>
#include <thread>
#include <usearch/index_dense.hpp>

#include "backports/algorithm.h"
#include "util/Exception.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
namespace uu = unum::usearch;

// We key the usearch index by the raw 64-bit `ValueId` bits, so the key type
// must be `uint64_t` (the default `index_dense_t` uses a signed key, which
// would misbehave for `ValueId`s whose datatype bits set the high bit).
using DenseIndex = uu::index_dense_gt<std::uint64_t>;

// How often the exact-search loops call the interrupt callback. Frequent enough
// for sub-second cancellation latency, rare enough to not show up in profiles.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// Map our enums to the corresponding usearch enum values.
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

uu::scalar_kind_t toUsearchScalar(VectorScalar s) {
  switch (s) {
    case VectorScalar::F32:
      return uu::scalar_kind_t::f32_k;
    case VectorScalar::F16:
      return uu::scalar_kind_t::f16_k;
    case VectorScalar::I8:
      return uu::scalar_kind_t::i8_k;
  }
  return uu::scalar_kind_t::f32_k;
}
}  // namespace

// All heavy members (mmaped vectors + usearch) live here so that neither
// usearch nor `MmapVector` leak into the public header.
struct VectorIndex::Impl {
  VectorIndexMetadata meta_;
  ad_utility::MmapVectorView<uint64_t> keys_;  // ascending entity ids
  ad_utility::MmapVectorView<float> data_;     // row-major, stride = dimensions
  std::optional<uu::metric_punned_t> metric_;  // shared by exact + HNSW
  std::optional<DenseIndex> hnsw_;             // present iff meta_.hasHnsw_

  size_t dim() const { return meta_.config_.dimensions_; }

  // Pointer to the start of row `i` in the flat float store.
  const float* rowPtr(size_t i) const { return data_.data() + i * dim(); }

  // Row index of `entity`, or nullopt if it has no vector here.
  std::optional<size_t> rowOf(Id entity) const {
    uint64_t id = entity.getBits();
    auto it = ql::ranges::lower_bound(keys_, id);
    if (it == keys_.end() || *it != id) {
      return std::nullopt;
    }
    return static_cast<size_t>(it - keys_.begin());
  }

  // Distance between the query and row `i`, using the (punned) index metric so
  // that exact and HNSW distances are identical.
  float distanceToRow(const float* query, size_t i) const {
    return static_cast<float>(
        metric_.value()(reinterpret_cast<const char*>(query),
                        reinterpret_cast<const char*>(rowPtr(i))));
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

  // 2. Flat store (memory-mapped, random access pattern for scattered lookups).
  auto openMmap = [&](auto& view, const std::string& path) {
    try {
      view.open(path, ad_utility::AccessPattern::Random);
    } catch (const std::exception& e) {
      AD_THROW("Could not open the file " + path + " of vector index \"" +
               name + "\": " + e.what());
    }
  };
  openMmap(impl.keys_, vectorKeysFile(basename, name));
  openMmap(impl.data_, vectorDataFile(basename, name));
  if (impl.keys_.size() != impl.meta_.numVectors_) {
    AD_THROW("Vector index \"" + name +
             "\": the number of keys on disk does not match the metadata. The "
             "index files are likely from an interrupted build; please rebuild "
             "the vector index.");
  }
  if (impl.data_.size() !=
      impl.meta_.numVectors_ * impl.meta_.config_.dimensions_) {
    AD_THROW("Vector index \"" + name +
             "\": the data size on disk does not match keys * dimensions. The "
             "index files are likely from an interrupted build; please rebuild "
             "the vector index.");
  }

  // 3. The shared metric.
  impl.metric_.emplace(impl.meta_.config_.dimensions_,
                       toUsearchMetric(impl.meta_.config_.metric_),
                       toUsearchScalar(impl.meta_.config_.scalar_));

  // 4. The optional HNSW index, memory-mapped read-only via usearch `view`.
  if (impl.meta_.hasHnsw_) {
    auto result =
        DenseIndex::make(vectorHnswFile(basename, name).c_str(), /*view=*/true);
    if (!result) {
      AD_THROW("Could not memory-map the HNSW file for vector index \"" + name +
               "\": " + result.error.what());
    }
    impl.hnsw_.emplace(std::move(result.index));
    // `view()` restores only what is in the usearch file header, which does
    // NOT include the search-expansion parameter -- apply the configured value
    // (otherwise every search silently runs at usearch's default ef).
    impl.hnsw_->change_expansion_search(
        impl.meta_.config_.hnswExpansionSearch_);
    // `view()` sizes the search-context pool to `hardware_concurrency()`, and
    // usearch FAILS (rather than queues) a search when the pool is exhausted.
    // QLever's query-thread count (`-j`) is user-configurable and may exceed
    // the core count, so reserve a generously larger pool up front.
    uu::index_limits_t limits;
    limits.members = impl.hnsw_->size();
    limits.threads_add = 0;
    limits.threads_search =
        std::max<std::size_t>(std::thread::hardware_concurrency(), 1) * 4;
    impl.hnsw_->try_reserve(limits);
  }
}

// ____________________________________________________________________________
const VectorIndexMetadata& VectorIndex::metadata() const {
  return impl_->meta_;
}
size_t VectorIndex::dimensions() const { return impl_->dim(); }
size_t VectorIndex::numVectors() const { return impl_->meta_.numVectors_; }
VectorMetric VectorIndex::metric() const {
  return impl_->meta_.config_.metric_;
}
bool VectorIndex::hasHnsw() const { return impl_->meta_.hasHnsw_; }

// ____________________________________________________________________________
std::optional<ql::span<const float>> VectorIndex::getVector(Id entity) const {
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  return ql::span<const float>{impl_->rowPtr(row.value()), impl_->dim()};
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
  if (k == 0 || impl.meta_.numVectors_ == 0) {
    return {};
  }
  TopK top{k};
  size_t sinceCheck = 0;
  auto consider = [&](size_t row, uint64_t id) {
    if (checkInterrupt && ++sinceCheck == CHECK_INTERRUPT_PERIOD) {
      sinceCheck = 0;
      checkInterrupt();
    }
    float dist = impl.distanceToRow(query.data(), row);
    if (!maxDistance.has_value() || dist <= maxDistance.value()) {
      top.offer(dist, id);
    }
  };
  if (!candidates.has_value()) {
    for (size_t i = 0; i < impl.meta_.numVectors_; ++i) {
      consider(i, impl.keys_[i]);
    }
  } else {
    // NOTE: an empty candidate list deliberately yields an empty result -- the
    // caller has already restricted the search space to nothing.
    for (Id c : candidates.value()) {
      if (auto row = impl.rowOf(c); row.has_value()) {
        consider(row.value(), c.getBits());
      }
    }
  }
  return top.sorted();
}

// ____________________________________________________________________________
std::vector<ScoredEntity> VectorIndex::searchHnsw(
    ql::span<const float> query, size_t k,
    std::optional<float> maxDistance) const {
  const auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.hnsw_.has_value(),
                    "searchHnsw called but this index has no HNSW structure.");
  if (query.size() != impl.dim()) {
    AD_THROW("The query vector has dimension " + std::to_string(query.size()) +
             ", but vector index \"" + impl.meta_.config_.name_ +
             "\" has dimension " + std::to_string(impl.dim()) + ".");
  }
  if (k == 0) {
    return {};
  }
  auto result = impl.hnsw_.value().search(query.data(), k);
  if (!result) {
    AD_THROW("HNSW search on vector index \"" + impl.meta_.config_.name_ +
             "\" failed: " + result.error.what());
  }
  using KeyT = DenseIndex::vector_key_t;
  using DistT = DenseIndex::distance_t;
  std::vector<KeyT> keys(k);
  std::vector<DistT> dists(k);
  size_t count = result.dump_to(keys.data(), dists.data());
  std::vector<ScoredEntity> out;
  out.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    float dist = static_cast<float>(dists[i]);
    if (maxDistance.has_value() && dist > maxDistance.value()) {
      continue;
    }
    out.push_back(
        ScoredEntity{Id::fromBits(static_cast<uint64_t>(keys[i])), dist});
  }
  return out;
}

}  // namespace qlever::vector
