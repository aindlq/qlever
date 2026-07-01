// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

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
constexpr size_t DIM = 16;
constexpr size_t NUM_VECTORS = 200;

// A per-test unique path prefix, so that concurrently running tests (ctest -j)
// never race on shared filenames.
std::string uniqueTmpBasename() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  return (std::filesystem::temp_directory_path() /
          (std::string{"qlever-vectest-"} + info->test_suite_name() + "-" +
           info->name()))
      .string();
}

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
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    d.rawIds.push_back(1000 + i * 7);
    std::vector<float> v(DIM);
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
  b.basename = uniqueTmpBasename();
  b.data = makeData();
  VectorIndexConfig cfg;
  cfg.name_ = b.name;
  cfg.dimensions_ = DIM;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = withHnsw;
  cfg.hnswExpansionSearch_ = 200;  // high recall for the test
  VectorIndexBuilder builder{b.basename, cfg};
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    builder.add(b.data.id(i), b.data.vecs[i]);
  }
  auto meta = builder.build();
  EXPECT_EQ(meta.numVectors_, NUM_VECTORS);
  EXPECT_EQ(meta.hasHnsw_, withHnsw);
  return b;
}

void cleanup(const BuiltIndex& b) {
  for (auto* suffix : {".meta", ".keys", ".data", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(b.basename + ".vec." + b.name + suffix, ec);
  }
}

// Write a minimal valid .npy (little-endian f32, C-order, shape (n,d)).
// `version` 1 writes a v1.0 header (2-byte length), 2 a v2.0 header (4-byte).
void writeNpy(const std::string& path, size_t n, size_t d,
              const std::vector<float>& data, int version = 1) {
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     std::to_string(n) + ", " + std::to_string(d) + "), }";
  size_t lenBytes = version == 1 ? 2 : 4;
  size_t base = 8 + lenBytes + dict.size() + 1;
  size_t pad = (64 - (base % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  std::ofstream out{path, std::ios::binary};
  out.write("\x93NUMPY", 6);
  char ver[2] = {static_cast<char>(version), 0};
  out.write(ver, 2);
  uint32_t hlen = static_cast<uint32_t>(dict.size());
  for (size_t i = 0; i < lenBytes; ++i) {
    char byte = static_cast<char>((hlen >> (8 * i)) & 0xff);
    out.write(&byte, 1);
  }
  out.write(dict.data(), dict.size());
  out.write(reinterpret_cast<const char*>(data.data()), n * d * sizeof(float));
}

// Write `iris` (one per line) to `path`.
void writeIris(const std::string& path, const TestData& data, size_t count) {
  std::ofstream out{path};
  for (size_t i = 0; i < count; ++i) {
    out << "<http://ex/" << data.rawIds[i % data.rawIds.size()] << ">\n";
  }
}
}  // namespace

// _____________________________________________________________________________
TEST(VectorIndex, buildOpenAndMetadata) {
  auto b = buildTmp(/*withHnsw=*/true);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.dimensions(), DIM);
  EXPECT_EQ(idx.numVectors(), NUM_VECTORS);
  EXPECT_EQ(idx.metric(), VectorMetric::Cosine);
  EXPECT_TRUE(idx.hasHnsw());
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndex, getVectorRoundTrip) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  auto v = idx.getVector(b.data.id(42));
  ASSERT_TRUE(v.has_value());
  ASSERT_EQ(v->size(), DIM);
  for (size_t j = 0; j < DIM; ++j) {
    EXPECT_FLOAT_EQ((*v)[j], b.data.vecs[42][j]);
  }
  EXPECT_FALSE(idx.getVector(mkId(999999)).has_value());
  cleanup(b);
}

// _____________________________________________________________________________
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
  // k larger than the index size gracefully returns everything.
  auto all = idx.searchExact(b.data.vecs[10], NUM_VECTORS + 50);
  EXPECT_EQ(all.size(), NUM_VECTORS);
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndex, exactSearchOverCandidateSubset) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> subset;
  std::set<uint64_t> subsetBits;
  for (size_t i = 0; i < NUM_VECTORS; i += 5) {
    subset.push_back(b.data.id(i));
    subsetBits.insert(b.data.id(i).getBits());
  }
  auto res = idx.searchExact(b.data.vecs[3], 10, ql::span<const Id>{subset});
  ASSERT_FALSE(res.empty());
  for (const auto& r : res) {
    EXPECT_TRUE(subsetBits.contains(r.entity_.getBits()));
  }
  // The results must be the k nearest WITHIN the subset: recompute by brute
  // force over the subset and compare the entity order.
  auto full = idx.searchExact(b.data.vecs[3], NUM_VECTORS);
  std::vector<uint64_t> expected;
  for (const auto& r : full) {
    if (subsetBits.contains(r.entity_.getBits())) {
      expected.push_back(r.entity_.getBits());
    }
    if (expected.size() == res.size()) break;
  }
  for (size_t i = 0; i < res.size(); ++i) {
    EXPECT_EQ(res[i].entity_.getBits(), expected[i]);
  }
  cleanup(b);
}

// _____________________________________________________________________________
// Regression test: an EMPTY candidate list must yield an empty result, not
// fall back to searching the whole index (the restriction matched nothing).
TEST(VectorIndex, emptyCandidateListYieldsEmptyResult) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> empty;
  auto res = idx.searchExact(b.data.vecs[0], 10, ql::span<const Id>{empty});
  EXPECT_TRUE(res.empty());
  // Candidates that all lack a vector likewise yield an empty result.
  std::vector<Id> unknown{mkId(999999), mkId(888888)};
  res = idx.searchExact(b.data.vecs[0], 10, ql::span<const Id>{unknown});
  EXPECT_TRUE(res.empty());
  // `nullopt` (the default) searches the whole index.
  EXPECT_FALSE(idx.searchExact(b.data.vecs[0], 10).empty());
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndex, maxDistanceFiltersResults) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  // With maxDistance 0 (and floating point fuzz), only the query point itself
  // remains.
  auto res = idx.searchExact(b.data.vecs[7], 10, std::nullopt, 1e-5f);
  ASSERT_EQ(res.size(), 1u);
  EXPECT_EQ(res.front().entity_, b.data.id(7));
  cleanup(b);
}

// _____________________________________________________________________________
// Duplicate entities in the input must not abort the build (usearch rejects
// duplicate keys); the first vector of each entity wins.
TEST(VectorIndex, duplicateEntitiesAreDeduplicated) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "dup";
  cfg.dimensions_ = 4;
  cfg.buildHnsw_ = true;
  VectorIndexBuilder builder{basename, cfg};
  std::vector<float> first{1.f, 0.f, 0.f, 0.f};
  std::vector<float> second{0.f, 1.f, 0.f, 0.f};
  std::vector<float> other{0.f, 0.f, 1.f, 0.f};
  builder.add(mkId(1), first);
  builder.add(mkId(1), second);  // duplicate -> dropped, first wins
  builder.add(mkId(2), other);
  auto meta = builder.build();
  EXPECT_EQ(meta.numVectors_, 2u);
  VectorIndex idx;
  idx.open(basename, "dup");
  auto v = idx.getVector(mkId(1));
  ASSERT_TRUE(v.has_value());
  EXPECT_FLOAT_EQ((*v)[0], 1.f);
  EXPECT_FLOAT_EQ((*v)[1], 0.f);
  for (auto* suffix : {".meta", ".keys", ".data", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.dup" + suffix, ec);
  }
}

// _____________________________________________________________________________
// The build must be atomic: rebuilding over an existing index either fully
// succeeds or leaves the previous files untouched (no `.meta`/`.keys` skew).
TEST(VectorIndex, rebuildOverExistingIndexIsAtomic) {
  auto b = buildTmp(/*withHnsw=*/false);
  // Rebuild with different content over the same basename.
  VectorIndexConfig cfg;
  cfg.name_ = b.name;
  cfg.dimensions_ = DIM;
  cfg.buildHnsw_ = false;
  VectorIndexBuilder builder{b.basename, cfg};
  builder.add(b.data.id(0), b.data.vecs[0]);
  builder.build();
  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.numVectors(), 1u);
  // No `.tmp` files are left behind.
  for (auto* suffix : {".meta.tmp", ".keys.tmp", ".data.tmp", ".hnsw.tmp"}) {
    EXPECT_FALSE(
        std::filesystem::exists(b.basename + ".vec." + b.name + suffix));
  }
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndex, openRejectsVersionMismatchAndCorruptMeta) {
  auto b = buildTmp(/*withHnsw=*/false);
  const std::string metaPath = b.basename + ".vec." + b.name + ".meta";
  // Patch the version in the metadata.
  {
    std::ifstream in{metaPath};
    std::string json((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
    auto pos = json.find("\"version\"");
    ASSERT_NE(pos, std::string::npos);
    json.replace(json.find(':', pos) + 1,
                 json.find_first_of(",}", pos) - json.find(':', pos) - 1,
                 " 999");
    std::ofstream out{metaPath};
    out << json;
  }
  VectorIndex idx;
  EXPECT_THROW(idx.open(b.basename, b.name), std::exception);
  // Corrupt metadata entirely.
  {
    std::ofstream out{metaPath};
    out << "this is not json";
  }
  EXPECT_THROW(idx.open(b.basename, b.name), std::exception);
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndex, npyInputReader) {
  auto data = makeData();
  std::string base = uniqueTmpBasename();
  std::string npy = base + ".npy";
  std::string iris = base + ".iris";
  std::vector<float> flat;
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    flat.insert(flat.end(), data.vecs[i].begin(), data.vecs[i].end());
  }
  writeIris(iris, data, NUM_VECTORS);
  writeNpy(npy, NUM_VECTORS, DIM, flat);

  NpyVectorInputReader reader{npy, iris};
  EXPECT_EQ(reader.dimensions(), DIM);
  EXPECT_EQ(reader.numRows(), NUM_VECTORS);
  std::string iri;
  std::vector<float> v;
  size_t count = 0;
  while (reader.next(iri, v)) {
    ASSERT_EQ(v.size(), DIM);
    EXPECT_EQ(iri, "<http://ex/" + std::to_string(data.rawIds[count]) + ">");
    for (size_t j = 0; j < DIM; ++j) EXPECT_FLOAT_EQ(v[j], data.vecs[count][j]);
    ++count;
  }
  EXPECT_EQ(count, NUM_VECTORS);
  std::error_code ec;
  std::filesystem::remove(npy, ec);
  std::filesystem::remove(iris, ec);
}

// _____________________________________________________________________________
TEST(VectorIndex, npyInputReaderV2HeaderAndErrors) {
  auto data = makeData();
  std::string base = uniqueTmpBasename();
  std::string npy = base + ".npy";
  std::string iris = base + ".iris";
  std::vector<float> flat;
  for (size_t i = 0; i < 10; ++i) {
    flat.insert(flat.end(), data.vecs[i].begin(), data.vecs[i].end());
  }

  // A v2.0 header (4-byte header length) parses fine.
  writeIris(iris, data, 10);
  writeNpy(npy, 10, DIM, flat, /*version=*/2);
  {
    NpyVectorInputReader reader{npy, iris};
    EXPECT_EQ(reader.numRows(), 10u);
    EXPECT_EQ(reader.dimensions(), DIM);
  }

  // Truncated float data -> error at the affected row, not garbage.
  writeNpy(npy, 10, DIM, flat);
  std::filesystem::resize_file(npy, std::filesystem::file_size(npy) - 2 * DIM);
  {
    NpyVectorInputReader reader{npy, iris};
    std::string iri;
    std::vector<float> v;
    EXPECT_THROW(
        {
          while (reader.next(iri, v)) {
          }
        },
        std::exception);
  }

  // IRI list shorter than the matrix -> error.
  writeNpy(npy, 10, DIM, flat);
  writeIris(iris, data, 5);
  {
    NpyVectorInputReader reader{npy, iris};
    std::string iri;
    std::vector<float> v;
    EXPECT_THROW(
        {
          while (reader.next(iri, v)) {
          }
        },
        std::exception);
  }

  // IRI list with MORE (non-empty) lines than the matrix -> error (a silent
  // mismatch would attach vectors to the wrong entities).
  writeIris(iris, data, 15);
  {
    NpyVectorInputReader reader{npy, iris};
    std::string iri;
    std::vector<float> v;
    EXPECT_THROW(
        {
          while (reader.next(iri, v)) {
          }
        },
        std::exception);
  }

  std::error_code ec;
  std::filesystem::remove(npy, ec);
  std::filesystem::remove(iris, ec);
}

// _____________________________________________________________________________
TEST(VectorIndex, hnswMatchesExactReasonably) {
  auto b = buildTmp(/*withHnsw=*/true);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  // The configured search expansion must survive the save/view round trip
  // (usearch's file header does not store it).
  EXPECT_EQ(idx.metadata().config_.hnswExpansionSearch_, 200u);
  size_t hits = 0, total = 0;
  for (size_t q = 0; q < NUM_VECTORS; q += 13) {
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
