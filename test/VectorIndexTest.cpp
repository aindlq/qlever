// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <set>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorInputReader.h"

using namespace qlever::vector;

namespace {
constexpr size_t kDim = 16;
constexpr size_t kN = 200;

// Turn a small integer into a realistic entity `Id` (a vocabulary index).
Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

// Deterministic random unit vectors, plus a non-contiguous id space so that
// "entity id" and "row index" are clearly different.
struct TestData {
  std::vector<uint64_t> rawIds;  // small ints used to form Ids and IRIs
  std::vector<std::vector<float>> vecs;
  Id id(size_t i) const { return mkId(rawIds[i]); }
};

TestData makeData() {
  std::mt19937 rng{12345};
  std::normal_distribution<float> g{0.f, 1.f};
  TestData d;
  for (size_t i = 0; i < kN; ++i) {
    d.rawIds.push_back(1000 + i * 7);
    std::vector<float> v(kDim);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    d.vecs.push_back(std::move(v));
  }
  return d;
}

struct BuiltIndex {
  std::string basename;
  std::string name = "clip";
  TestData data;
};

BuiltIndex buildTmp(bool withHnsw) {
  BuiltIndex b;
  b.basename = (std::filesystem::temp_directory_path() /
                ("qlever-vectest-" + std::to_string(withHnsw)))
                   .string();
  b.data = makeData();
  VectorIndexConfig cfg;
  cfg.name = b.name;
  cfg.dimensions = kDim;
  cfg.metric = VectorMetric::Cosine;
  cfg.buildHnsw = withHnsw;
  cfg.hnswExpansionSearch = 200;  // high recall for the test
  VectorIndexBuilder builder{b.basename, cfg};
  for (size_t i = 0; i < kN; ++i) {
    builder.add(b.data.id(i), b.data.vecs[i]);
  }
  auto meta = builder.build();
  EXPECT_EQ(meta.numVectors, kN);
  EXPECT_EQ(meta.hasHnsw, withHnsw);
  return b;
}

void cleanup(const BuiltIndex& b) {
  for (auto* suffix : {".meta", ".keys", ".data", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(b.basename + ".vec." + b.name + suffix, ec);
  }
}
}  // namespace

TEST(VectorIndex, buildOpenAndMetadata) {
  auto b = buildTmp(/*withHnsw=*/true);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.dimensions(), kDim);
  EXPECT_EQ(idx.numVectors(), kN);
  EXPECT_EQ(idx.metric(), VectorMetric::Cosine);
  EXPECT_TRUE(idx.hasHnsw());
  cleanup(b);
}

TEST(VectorIndex, getVectorRoundTrip) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  auto v = idx.getVector(b.data.id(42));
  ASSERT_TRUE(v.has_value());
  ASSERT_EQ(v->size(), kDim);
  for (size_t j = 0; j < kDim; ++j) {
    EXPECT_FLOAT_EQ((*v)[j], b.data.vecs[42][j]);
  }
  EXPECT_FALSE(idx.getVector(mkId(999999)).has_value());
  cleanup(b);
}

TEST(VectorIndex, exactSearchNearestIsSelf) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  auto res = idx.searchExact(b.data.vecs[10], 5);
  ASSERT_FALSE(res.empty());
  EXPECT_EQ(res.front().entity_, b.data.id(10));
  EXPECT_NEAR(res.front().distance_, 0.f, 1e-4);
  for (size_t i = 1; i < res.size(); ++i) {
    EXPECT_GE(res[i].distance_, res[i - 1].distance_);
  }
  cleanup(b);
}

TEST(VectorIndex, exactSearchOverCandidateSubset) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> subset;
  std::set<uint64_t> subsetBits;
  for (size_t i = 0; i < kN; i += 5) {
    subset.push_back(b.data.id(i));
    subsetBits.insert(b.data.id(i).getBits());
  }
  auto res = idx.searchExact(b.data.vecs[3], 10, subset);
  ASSERT_FALSE(res.empty());
  for (const auto& r : res) {
    EXPECT_TRUE(subsetBits.contains(r.entity_.getBits()));
  }
  cleanup(b);
}

// Write a minimal valid v1.0 .npy (little-endian f32, C-order, shape (n,d)).
static void writeNpy(const std::string& path, size_t n, size_t d,
                     const std::vector<float>& data) {
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     std::to_string(n) + ", " + std::to_string(d) + "), }";
  size_t base = 10 + dict.size() + 1;
  size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  uint16_t hlen = static_cast<uint16_t>(dict.size());
  std::ofstream out{path, std::ios::binary};
  out.write("\x93NUMPY", 6);
  char ver[2] = {1, 0};
  out.write(ver, 2);
  char lb[2] = {static_cast<char>(hlen & 0xff),
                static_cast<char>((hlen >> 8) & 0xff)};
  out.write(lb, 2);
  out.write(dict.data(), dict.size());
  out.write(reinterpret_cast<const char*>(data.data()), n * d * sizeof(float));
}

TEST(VectorIndex, npyInputReader) {
  auto data = makeData();
  auto dir = std::filesystem::temp_directory_path();
  std::string npy = (dir / "qlever-vectest.npy").string();
  std::string iris = (dir / "qlever-vectest.iris").string();
  std::vector<float> flat;
  std::ofstream irisOut{iris};
  for (size_t i = 0; i < kN; ++i) {
    irisOut << "<http://ex/" << data.rawIds[i] << ">\n";
    flat.insert(flat.end(), data.vecs[i].begin(), data.vecs[i].end());
  }
  irisOut.close();
  writeNpy(npy, kN, kDim, flat);

  NpyVectorInputReader reader{npy, iris};
  EXPECT_EQ(reader.dimensions(), kDim);
  EXPECT_EQ(reader.numRows(), kN);
  std::string iri;
  std::vector<float> v;
  size_t count = 0;
  while (reader.next(iri, v)) {
    ASSERT_EQ(v.size(), kDim);
    EXPECT_EQ(iri, "<http://ex/" + std::to_string(data.rawIds[count]) + ">");
    for (size_t j = 0; j < kDim; ++j) EXPECT_FLOAT_EQ(v[j], data.vecs[count][j]);
    ++count;
  }
  EXPECT_EQ(count, kN);
  std::error_code ec;
  std::filesystem::remove(npy, ec);
  std::filesystem::remove(iris, ec);
}

TEST(VectorIndex, hnswMatchesExactReasonably) {
  auto b = buildTmp(/*withHnsw=*/true);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  size_t hits = 0, total = 0;
  for (size_t q = 0; q < kN; q += 13) {
    auto exact = idx.searchExact(b.data.vecs[q], 10);
    auto hnsw = idx.searchHnsw(b.data.vecs[q], 10);
    ASSERT_FALSE(hnsw.empty());
    EXPECT_EQ(hnsw.front().entity_, b.data.id(q));
    std::set<uint64_t> truth;
    for (const auto& e : exact) truth.insert(e.entity_.getBits());
    for (const auto& h : hnsw)
      if (truth.contains(h.entity_.getBits())) ++hits;
    total += exact.size();
  }
  EXPECT_GE(static_cast<double>(hits) / total, 0.9);
  cleanup(b);
}
