// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorIndex.h"

#include <algorithm>
#include <fstream>
#include <queue>

#include <usearch/index_dense.hpp>

#include "util/Exception.h"
#include "util/MmapVector.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
namespace uu = unum::usearch;

// We key the usearch index by the raw 64-bit `ValueId` bits, so the key type
// must be `uint64_t` (the default `index_dense_t` uses a signed key, which would
// misbehave for `ValueId`s whose datatype bits set the high bit).
using DenseIndex = uu::index_dense_gt<std::uint64_t>;

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
  std::optional<DenseIndex> hnsw_;       // present iff meta_.hasHnsw

  size_t dim() const { return meta_.config.dimensions; }

  // Pointer to the start of row `i` in the flat float store.
  const float* rowPtr(size_t i) const { return data_.data() + i * dim(); }

  // Row index of `entity`, or nullopt if it has no vector here.
  std::optional<size_t> rowOf(Id entity) const {
    uint64_t id = entity.getBits();
    auto begin = keys_.begin();
    auto end = keys_.end();
    auto it = std::lower_bound(begin, end, id);
    if (it == end || *it != id) {
      return std::nullopt;
    }
    return static_cast<size_t>(it - begin);
  }

  // Distance between the query and row `i`, using the (punned) index metric so
  // that exact and HNSW distances are identical.
  float distanceToRow(const float* query, size_t i) const {
    return static_cast<float>(metric_.value()(
        reinterpret_cast<const char*>(query),
        reinterpret_cast<const char*>(rowPtr(i))));
  }
};

VectorIndex::VectorIndex() : impl_{std::make_unique<Impl>()} {}
VectorIndex::~VectorIndex() = default;
VectorIndex::VectorIndex(VectorIndex&&) noexcept = default;
VectorIndex& VectorIndex::operator=(VectorIndex&&) noexcept = default;

void VectorIndex::open(const std::string& basename, const std::string& name) {
  auto& impl = *impl_;

  // 1. Metadata.
  std::ifstream metaIn{vectorMetaFile(basename, name)};
  AD_CONTRACT_CHECK(metaIn.is_open(),
                    "Could not open vector index metadata file ",
                    vectorMetaFile(basename, name));
  nlohmann::json j;
  metaIn >> j;
  impl.meta_ = j.get<VectorIndexMetadata>();
  AD_CONTRACT_CHECK(impl.meta_.version == VECTOR_INDEX_VERSION,
                    "Vector index ", name, " has on-disk version ",
                    impl.meta_.version, " but this binary expects ",
                    VECTOR_INDEX_VERSION, ". Please rebuild the index.");

  // 2. Flat store (memory-mapped, random access pattern for scattered lookups).
  impl.keys_.open(vectorKeysFile(basename, name),
                  ad_utility::AccessPattern::Random);
  impl.data_.open(vectorDataFile(basename, name),
                  ad_utility::AccessPattern::Random);
  AD_CONTRACT_CHECK(impl.keys_.size() == impl.meta_.numVectors,
                    "Vector index ", name,
                    ": number of keys does not match metadata.");
  AD_CONTRACT_CHECK(
      impl.data_.size() == impl.meta_.numVectors * impl.meta_.config.dimensions,
      "Vector index ", name, ": data size does not match keys * dimensions.");

  // 3. The shared metric.
  impl.metric_.emplace(impl.meta_.config.dimensions,
                       toUsearchMetric(impl.meta_.config.metric),
                       toUsearchScalar(impl.meta_.config.scalar));

  // 4. The optional HNSW index, memory-mapped read-only via usearch `view`.
  if (impl.meta_.hasHnsw) {
    auto result = DenseIndex::make(
        vectorHnswFile(basename, name).c_str(), /*view=*/true);
    AD_CONTRACT_CHECK(static_cast<bool>(result),
                      "Could not memory-map the HNSW file for vector index ",
                      name, ": ", result.error.what());
    impl.hnsw_.emplace(std::move(result.index));
  }
}

const VectorIndexMetadata& VectorIndex::metadata() const {
  return impl_->meta_;
}
size_t VectorIndex::dimensions() const { return impl_->dim(); }
size_t VectorIndex::numVectors() const { return impl_->meta_.numVectors; }
VectorMetric VectorIndex::metric() const { return impl_->meta_.config.metric; }
bool VectorIndex::hasHnsw() const { return impl_->meta_.hasHnsw; }

std::optional<std::span<const float>> VectorIndex::getVector(
    Id entity) const {
  auto row = impl_->rowOf(entity);
  if (!row.has_value()) {
    return std::nullopt;
  }
  return std::span<const float>{impl_->rowPtr(row.value()), impl_->dim()};
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
    std::reverse(out.begin(), out.end());
    return out;
  }

 private:
  size_t k_;
  std::priority_queue<std::pair<float, uint64_t>> heap_;
};
}  // namespace

std::vector<ScoredEntity> VectorIndex::searchExact(
    std::span<const float> query, size_t k,
    std::span<const Id> candidates,
    std::optional<float> maxDistance) const {
  const auto& impl = *impl_;
  AD_CONTRACT_CHECK(query.size() == impl.dim(),
                    "Query vector dimension does not match the index.");
  if (k == 0 || impl.meta_.numVectors == 0) {
    return {};
  }
  TopK top{k};
  auto consider = [&](size_t row, uint64_t id) {
    float dist = impl.distanceToRow(query.data(), row);
    if (!maxDistance.has_value() || dist <= maxDistance.value()) {
      top.offer(dist, id);
    }
  };
  if (candidates.empty()) {
    for (size_t i = 0; i < impl.meta_.numVectors; ++i) {
      consider(i, impl.keys_[i]);
    }
  } else {
    for (Id c : candidates) {
      if (auto row = impl.rowOf(c); row.has_value()) {
        consider(row.value(), c.getBits());
      }
    }
  }
  return top.sorted();
}

std::vector<ScoredEntity> VectorIndex::searchHnsw(
    std::span<const float> query, size_t k,
    std::optional<float> maxDistance) const {
  const auto& impl = *impl_;
  AD_CONTRACT_CHECK(impl.hnsw_.has_value(),
                    "searchHnsw called but this index has no HNSW structure.");
  AD_CONTRACT_CHECK(query.size() == impl.dim(),
                    "Query vector dimension does not match the index.");
  if (k == 0) {
    return {};
  }
  auto result = impl.hnsw_.value().search(query.data(), k);
  AD_CONTRACT_CHECK(static_cast<bool>(result), "HNSW search failed: ",
                    result.error.what());
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
