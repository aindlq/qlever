// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorInputReader.h"
#include "util/json.h"

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
  std::string iri(size_t i) const {
    return "<http://ex/" + std::to_string(rawIds[i]) + ">";
  }
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
    builder.add(b.data.id(i), b.data.iri(i), b.data.vecs[i]);
  }
  auto meta = builder.build();
  EXPECT_EQ(meta.numVectors_, NUM_VECTORS);
  EXPECT_EQ(meta.hasHnsw_, withHnsw);
  return b;
}

void cleanup(const BuiltIndex& b) {
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
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

// Write `iris` (one per line) to `path`. Alternates between the iriref form
// `<...>` and a bare IRI with surrounding whitespace -- the reader accepts
// both and yields the trimmed bare form.
void writeIris(const std::string& path, const TestData& data, size_t count) {
  std::ofstream out{path};
  for (size_t i = 0; i < count; ++i) {
    uint64_t id = data.rawIds[i % data.rawIds.size()];
    if (i % 2 == 0) {
      out << "<http://ex/" << id << ">\n";
    } else {
      out << "  http://ex/" << id << " \t\n";
    }
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
// The `DistanceComputer` (the primitive behind `vec:distance`) must return, for
// every entity, exactly the distance that a brute-force `searchExact` reports
// (same punned metric), `NaN` for an entity with no vector, and throw on a
// dimension mismatch.
TEST(VectorIndex, distanceComputerMatchesSearchExact) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);

  const auto& query = b.data.vecs[10];
  // Ground truth: the distance of every entity to `query`.
  auto scored = idx.searchExact(query, NUM_VECTORS);
  ASSERT_EQ(scored.size(), NUM_VECTORS);
  std::map<uint64_t, float> expected;
  for (const auto& s : scored) {
    expected[s.entity_.getBits()] = s.distance_;
  }

  auto computer = idx.makeDistanceComputer(query);
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    Id e = b.data.id(i);
    ASSERT_TRUE(expected.count(e.getBits()));
    EXPECT_FLOAT_EQ(computer(e), expected[e.getBits()]) << "entity " << i;
  }
  // Self-distance is (near) zero for cosine.
  EXPECT_NEAR(computer(b.data.id(10)), 0.f, 1e-4);
  // An entity with no vector in the index -> NaN.
  EXPECT_TRUE(std::isnan(computer(mkId(999999))));

  // A dimension mismatch throws (query too short).
  std::vector<float> tooShort(DIM - 1, 0.f);
  EXPECT_ANY_THROW(idx.makeDistanceComputer(tooShort));
  cleanup(b);
}

// _____________________________________________________________________________
// The by-entity `DistanceComputer` uses a stored vector as the query point and
// must agree with `searchExactByEntity`; a missing entity yields `nullopt`.
TEST(VectorIndex, distanceComputerByEntity) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);

  auto scored = idx.searchExactByEntity(b.data.id(42), NUM_VECTORS);
  ASSERT_EQ(scored.size(), NUM_VECTORS);
  std::map<uint64_t, float> expected;
  for (const auto& s : scored) {
    expected[s.entity_.getBits()] = s.distance_;
  }

  auto computer = idx.makeDistanceComputerByEntity(b.data.id(42));
  ASSERT_TRUE(computer.has_value());
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    Id e = b.data.id(i);
    EXPECT_FLOAT_EQ((*computer)(e), expected[e.getBits()]) << "entity " << i;
  }
  EXPECT_NEAR((*computer)(b.data.id(42)), 0.f, 1e-4);
  EXPECT_TRUE(std::isnan((*computer)(mkId(999999))));

  // No vector for this entity -> no computer.
  EXPECT_FALSE(idx.makeDistanceComputerByEntity(mkId(999999)).has_value());
  cleanup(b);
}

// _____________________________________________________________________________
// The sorted-input fast path (`gatherSortedDistances`, the merge-walk behind
// `vec:distance` when the entity column is the leading sort key) must produce
// results BIT-IDENTICAL to the per-row primitive `(*computer)(entity)`, for an
// ascending column that mixes live members, interleaved non-members, and
// consecutive DUPLICATE entities, and that exceeds the parallel threshold.
// This simultaneously exercises `DistanceComputer::atRow`: every member's
// `out[i]` is `distanceToValueId(computer.atRow(row))`, so matching the
// reference proves `atRow(rowOf(id)) == (*computer)(id)`.
TEST(VectorIndex, gatherSortedDistancesIsBitIdenticalToPerRow) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);

  // A fixed query point (a stored unit vector, so distances vary and one
  // member sits at ~0).
  auto computer = idx.makeDistanceComputer(b.data.vecs[3]);

  // Build a strictly non-decreasing (by Id bits) column > the parallel
  // threshold. Member ids are `mkId(1000 + 7*i)`; every other raw value is a
  // NON-member. Consecutive duplicates are injected on both members and
  // non-members. Ids are ascending because `mkId` is monotone in the raw
  // value.
  std::vector<Id> column;
  bool sawMemberDup = false;
  bool sawNonMemberUndef = false;
  for (uint64_t v = 900; column.size() < 5000; ++v) {
    Id id = mkId(v);
    column.push_back(id);
    if (v % 4 == 0) {
      column.push_back(id);  // a consecutive duplicate
    }
    if (v % 40 == 0) {
      column.push_back(id);  // occasionally a triple
    }
  }
  ASSERT_GT(column.size(), 2048u);
  ASSERT_TRUE(std::is_sorted(column.begin(), column.end(), [](Id a, Id c) {
    return a.getBits() < c.getBits();
  }));

  // The serial per-row REFERENCE, computed only through the public per-row
  // primitive `(*computer)(entity)` + the shared distance->Id mapping.
  std::vector<Id> reference(column.size());
  for (size_t i = 0; i < column.size(); ++i) {
    reference[i] = qlever::vector::distanceToValueId(computer(column[i]));
    if (reference[i].isUndefined()) {
      sawNonMemberUndef = true;
    }
    if (i > 0 && column[i] == column[i - 1] && !reference[i].isUndefined()) {
      sawMemberDup = true;
    }
  }
  ASSERT_TRUE(sawMemberDup) << "the column must contain a duplicated member";
  ASSERT_TRUE(sawNonMemberUndef)
      << "the column must contain an interleaved non-member";

  // At this layer a non-member is genuinely UNDEF (no vocab/float-list to
  // resolve), matching the reference's NaN->UNDEF.
  auto undefOnMiss = [](size_t) { return Id::makeUndefined(); };
  auto neverCancel = [] { return false; };

  // Parallel walk (numThreads > 1 exercises the OpenMP chunked branch).
  std::vector<Id> parallelOut(column.size());
  idx.gatherSortedDistances(column, computer, undefOnMiss, neverCancel,
                            /*numThreads=*/4, parallelOut);
  // Serial walk (numThreads == 1 forces the serial branch).
  std::vector<Id> serialOut(column.size());
  idx.gatherSortedDistances(column, computer, undefOnMiss, neverCancel,
                            /*numThreads=*/1, serialOut);
  // A second parallel run, for determinism.
  std::vector<Id> parallelOut2(column.size());
  idx.gatherSortedDistances(column, computer, undefOnMiss, neverCancel,
                            /*numThreads=*/4, parallelOut2);

  for (size_t i = 0; i < column.size(); ++i) {
    ASSERT_EQ(parallelOut[i], reference[i]) << "parallel row " << i;
    ASSERT_EQ(serialOut[i], reference[i]) << "serial row " << i;
    ASSERT_EQ(parallelOut2[i], parallelOut[i]) << "determinism row " << i;
    // Consecutive duplicate ids resolve to identical results.
    if (i > 0 && column[i] == column[i - 1]) {
      ASSERT_EQ(parallelOut[i], parallelOut[i - 1]) << "dup row " << i;
    }
  }
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
// The merge-join gather helper: emits each matching candidate exactly once,
// in NON-DECREASING row order (the "monotonic gather" property that makes the
// `.data` reads sequential), and skips candidates with no row.
TEST(VectorIndex, mergeJoinRowmapMonotonicAndSkips) {
  // Ascending-by-id rowmap; `row_` is monotonic (identity in a fresh build).
  std::vector<IdRowPair> rowmap{{10, 0}, {20, 1}, {30, 2}, {40, 3}, {50, 4}};
  // Sorted candidates: 5/60 have no row, 20 is duplicated.
  std::vector<uint64_t> cands{5, 20, 20, 40, 60};
  std::vector<size_t> rows;
  std::vector<uint64_t> ids;
  mergeJoinRowmap(cands.begin(), cands.end(), rowmap.begin(), rowmap.end(),
                  [&](size_t row, uint64_t id) {
                    rows.push_back(row);
                    ids.push_back(id);
                  });
  EXPECT_EQ(ids, (std::vector<uint64_t>{20, 40}));  // matched, dup collapsed
  EXPECT_EQ(rows, (std::vector<size_t>{1, 3}));
  EXPECT_TRUE(std::is_sorted(rows.begin(), rows.end()));

  // Tombstone case: a remapped rowmap has monotonic-WITH-GAPS rows; the gather
  // is still non-decreasing.
  std::vector<IdRowPair> gapped{{11, 0}, {22, 3}, {33, 7}, {44, 9}};
  std::vector<uint64_t> cands2{22, 33, 44};
  std::vector<size_t> rows2;
  mergeJoinRowmap(cands2.begin(), cands2.end(), gapped.begin(), gapped.end(),
                  [&](size_t row, uint64_t) { rows2.push_back(row); });
  EXPECT_EQ(rows2, (std::vector<size_t>{3, 7, 9}));
  EXPECT_TRUE(std::is_sorted(rows2.begin(), rows2.end()));
}

// _____________________________________________________________________________
// A DENSE candidate set (all live entities) via the merge-join must return the
// exact same top-k as the unrestricted whole-index search.
TEST(VectorIndex, exactSearchDenseCandidatesMatchesFull) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> all;
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    all.push_back(b.data.id(i));
  }
  auto res = idx.searchExact(b.data.vecs[3], 10, ql::span<const Id>{all});
  auto full = idx.searchExact(b.data.vecs[3], 10);
  ASSERT_EQ(res.size(), full.size());
  for (size_t i = 0; i < res.size(); ++i) {
    EXPECT_EQ(res[i].entity_, full[i].entity_);
    EXPECT_EQ(res[i].distance_, full[i].distance_);
  }
  cleanup(b);
}

// _____________________________________________________________________________
// Candidates arriving UNSORTED (the merge sorts a local copy) yield exactly the
// same result as the same set sorted, and duplicates do not double-count.
TEST(VectorIndex, exactSearchUnsortedCandidatesStillCorrect) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> subset;
  for (size_t i = 0; i < NUM_VECTORS; i += 3) {
    subset.push_back(b.data.id(i));
  }
  auto sortedRes =
      idx.searchExact(b.data.vecs[7], 8, ql::span<const Id>{subset});
  // Shuffle and add duplicates; the result must be identical.
  std::vector<Id> shuffled = subset;
  shuffled.insert(shuffled.end(), subset.begin(), subset.begin() + 5);
  std::mt19937 rng{999};
  std::shuffle(shuffled.begin(), shuffled.end(), rng);
  auto shuffledRes =
      idx.searchExact(b.data.vecs[7], 8, ql::span<const Id>{shuffled});
  ASSERT_EQ(shuffledRes.size(), sortedRes.size());
  for (size_t i = 0; i < sortedRes.size(); ++i) {
    EXPECT_EQ(shuffledRes[i].entity_, sortedRes[i].entity_);
    EXPECT_EQ(shuffledRes[i].distance_, sortedRes[i].distance_);
  }
  cleanup(b);
}

// _____________________________________________________________________________
// A non-`VocabIndex` candidate id (here a double `Id`, whose bits are not a
// stored entity) is simply skipped by the merge; the other candidates are
// unaffected.
TEST(VectorIndex, exactSearchNonVocabIndexCandidateSkipped) {
  auto b = buildTmp(/*withHnsw=*/false);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  std::vector<Id> valid{b.data.id(1), b.data.id(2), b.data.id(3)};
  std::vector<Id> mixed = valid;
  mixed.push_back(Id::makeFromDouble(3.14));  // no vector, must be skipped
  mixed.push_back(Id::makeFromBool(true));    // ditto
  auto validRes = idx.searchExact(b.data.vecs[2], 5, ql::span<const Id>{valid});
  auto mixedRes = idx.searchExact(b.data.vecs[2], 5, ql::span<const Id>{mixed});
  ASSERT_EQ(mixedRes.size(), validRes.size());
  for (size_t i = 0; i < validRes.size(); ++i) {
    EXPECT_EQ(mixedRes[i].entity_, validRes[i].entity_);
    EXPECT_EQ(mixedRes[i].distance_, validRes[i].distance_);
  }
  cleanup(b);
}

// _____________________________________________________________________________
// The collation fingerprint set on the builder is persisted in the metadata,
// survives a JSON round-trip, and defaults to empty for a metadata that never
// carried it (back-compat).
TEST(VectorIndex, collationLocalePersistedInMetadata) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "col";
  cfg.dimensions_ = 4;
  cfg.buildHnsw_ = false;
  VectorIndexBuilder builder{basename, cfg};
  builder.setCollationLocale("en_US|nonIgnorable");
  builder.add(mkId(1), "<http://ex/1>", std::vector<float>{1, 0, 0, 0});
  builder.add(mkId(2), "<http://ex/2>", std::vector<float>{0, 1, 0, 0});
  auto meta = builder.build();
  EXPECT_EQ(meta.collationLocale_, "en_US|nonIgnorable");
  VectorIndex idx;
  idx.open(basename, "col");
  EXPECT_EQ(idx.metadata().collationLocale_, "en_US|nonIgnorable");
  // JSON round-trip preserves it; an absent key defaults to empty.
  nlohmann::json j = meta;
  EXPECT_EQ(j.get<VectorIndexMetadata>().collationLocale_,
            "en_US|nonIgnorable");
  j.erase("collationLocale");
  EXPECT_EQ(j.get<VectorIndexMetadata>().collationLocale_, std::string{});
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.col" + suffix, ec);
  }
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
  builder.add(mkId(1), "<http://ex/1>", first);
  builder.add(mkId(1), "<http://ex/1>", second);  // duplicate -> dropped
  builder.add(mkId(2), "<http://ex/2>", other);
  auto meta = builder.build();
  EXPECT_EQ(meta.numVectors_, 2u);
  VectorIndex idx;
  idx.open(basename, "dup");
  auto v = idx.getVector(mkId(1));
  ASSERT_TRUE(v.has_value());
  EXPECT_FLOAT_EQ((*v)[0], 1.f);
  EXPECT_FLOAT_EQ((*v)[1], 0.f);
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
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
  builder.add(b.data.id(0), b.data.iri(0), b.data.vecs[0]);
  builder.build();
  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.numVectors(), 1u);
  // No `.tmp` files are left behind.
  for (auto* suffix : {".meta.tmp", ".keys.tmp", ".rowmap.tmp", ".data.tmp",
                       ".iris.tmp", ".hnsw.tmp"}) {
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
    // Both the `<...>` lines and the whitespace-padded bare lines come out
    // as the trimmed bare IRI.
    EXPECT_EQ(iri, "http://ex/" + std::to_string(data.rawIds[count]));
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

// _____________________________________________________________________________
// f16 and i8 storage: half/quarter disk and page-cache footprint. `getVector`
// decodes to f32; searches work both with f32 queries (encoded once) and with
// stored-entity queries (raw stored bytes, no conversion); the HNSW graph is
// built directly over the quantized store.
TEST(VectorIndex, f16AndI8Storage) {
  auto data = makeData();  // unit vectors -- also in i8's expected [-1, 1]
  for (auto scalar : {VectorScalar::F16, VectorScalar::I8}) {
    std::string basename = uniqueTmpBasename() + "-" + toString(scalar);
    VectorIndexConfig cfg;
    cfg.name_ = "q";
    cfg.dimensions_ = DIM;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.scalar_ = scalar;
    cfg.buildHnsw_ = true;
    cfg.hnswExpansionSearch_ = 200;
    VectorIndexBuilder builder{basename, cfg};
    for (size_t i = 0; i < NUM_VECTORS; ++i) {
      builder.add(data.id(i), data.iri(i), data.vecs[i]);
    }
    auto meta = builder.build();
    EXPECT_EQ(meta.numVectors_, NUM_VECTORS);

    // The data file shrinks according to the scalar width (plus the small
    // trailer of the mmap format).
    EXPECT_LE(std::filesystem::file_size(basename + ".vec.q.data"),
              NUM_VECTORS * DIM * bytesPerScalar(scalar) + 4096);

    VectorIndex idx;
    idx.open(basename, "q");
    // Decode round trip within the quantization error.
    auto v = idx.getVector(data.id(7));
    ASSERT_TRUE(v.has_value());
    float tolerance = scalar == VectorScalar::F16 ? 5e-3f : 3e-2f;
    for (size_t j = 0; j < DIM; ++j) {
      EXPECT_NEAR((*v)[j], data.vecs[7][j], tolerance);
    }
    // f32 query (encoded to the storage scalar once): nearest is self.
    auto res = idx.searchExact(data.vecs[10], 3);
    ASSERT_FALSE(res.empty());
    EXPECT_EQ(res.front().entity_, data.id(10));
    EXPECT_NEAR(res.front().distance_, 0.f,
                scalar == VectorScalar::F16 ? 1e-3 : 5e-2);
    // Stored-entity query through the HNSW graph: nearest is self.
    auto hnswRes = idx.searchHnswByEntity(data.id(20), 3);
    ASSERT_FALSE(hnswRes.empty());
    EXPECT_EQ(hnswRes.front().entity_, data.id(20));
    // ... and over a candidate subset.
    std::vector<Id> subset{data.id(20), data.id(21), data.id(22)};
    auto sub =
        idx.searchExactByEntity(data.id(20), 2, ql::span<const Id>{subset});
    ASSERT_FALSE(sub.empty());
    EXPECT_EQ(sub.front().entity_, data.id(20));
    for (auto* suffix :
         {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
      std::error_code ec;
      std::filesystem::remove(basename + ".vec.q" + suffix, ec);
    }
  }
}

// _____________________________________________________________________________
// `i8` storage is rejected for metrics other than cosine (it normalizes).
TEST(VectorIndex, i8RejectsNonCosineMetric) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "q";
  cfg.dimensions_ = 4;
  cfg.scalar_ = VectorScalar::I8;
  cfg.metric_ = VectorMetric::L2Sq;
  // The builder itself allows it (the guard lives in the service spec parser);
  // this test documents that i8 + cosine builds and searches fine.
  cfg.metric_ = VectorMetric::Cosine;
  VectorIndexBuilder builder{basename, cfg};
  std::vector<float> v{1.f, 0.f, 0.f, 0.f};
  builder.add(mkId(1), "<http://ex/1>", v);
  builder.build();
  VectorIndex idx;
  idx.open(basename, "q");
  EXPECT_EQ(idx.searchExactByEntity(mkId(1), 1).front().entity_, mkId(1));
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.q" + suffix, ec);
  }
}

// _____________________________________________________________________________
// A huge `k` is clamped to the index size (no unbounded allocation).
TEST(VectorIndex, hugeKIsClamped) {
  auto b = buildTmp(/*withHnsw=*/true);
  VectorIndex idx;
  idx.open(b.basename, b.name);
  auto exact = idx.searchExact(b.data.vecs[0], size_t{1} << 40);
  EXPECT_EQ(exact.size(), NUM_VECTORS);
  auto hnsw = idx.searchHnsw(b.data.vecs[0], size_t{1} << 40);
  EXPECT_LE(hnsw.size(), NUM_VECTORS);
  EXPECT_GT(hnsw.size(), 0u);
  cleanup(b);
}

// _____________________________________________________________________________
// Helper: build an index with an arbitrary dimension and the `alignRows_` flag,
// returning the basename. Uses `dim` deliberately chosen so that the raw row
// byte length is NOT already a multiple of 64 (so padding is observable).
namespace {
struct AlignBuilt {
  std::string basename;
  std::string name = "al";
  std::vector<std::vector<float>> vecs;
  std::vector<Id> ids;
};

AlignBuilt buildAligned(size_t dim, bool alignRows, bool withHnsw,
                        const std::string& tag) {
  AlignBuilt b;
  b.basename = uniqueTmpBasename() + "-" + tag;
  std::mt19937 rng{9876};
  std::normal_distribution<float> g{0.f, 1.f};
  VectorIndexConfig cfg;
  cfg.name_ = b.name;
  cfg.dimensions_ = static_cast<uint32_t>(dim);
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = withHnsw;
  cfg.hnswExpansionSearch_ = 200;
  cfg.alignRows_ = alignRows;
  VectorIndexBuilder builder{b.basename, cfg};
  for (size_t i = 0; i < NUM_VECTORS; ++i) {
    std::vector<float> v(dim);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    Id id = mkId(1000 + i * 7);
    builder.add(id, "<http://ex/" + std::to_string(1000 + i * 7) + ">", v);
    b.vecs.push_back(std::move(v));
    b.ids.push_back(id);
  }
  builder.build();
  return b;
}

void cleanupBase(const std::string& basename, const std::string& name) {
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec." + name + suffix, ec);
  }
}
}  // namespace

// _____________________________________________________________________________
// `alignRows_` pads every row up to a 64-byte SIMD boundary (persisted as
// `rowStrideBytes_`), while leaving results bit-identical to an unpadded build.
TEST(VectorIndex, alignedRowsStridePaddingAndParity) {
  constexpr size_t dim = 10;  // raw row bytes = 40, not a multiple of 64
  constexpr size_t rawRowBytes = dim * sizeof(float);
  auto unaligned = buildAligned(dim, /*alignRows=*/false, false, "unal");
  auto aligned = buildAligned(dim, /*alignRows=*/true, false, "al");

  VectorIndex idxU;
  idxU.open(unaligned.basename, unaligned.name);
  VectorIndex idxA;
  idxA.open(aligned.basename, aligned.name);

  // Unpadded stride == raw row length; padded stride is the next multiple
  // of 64.
  EXPECT_EQ(idxU.metadata().rowStrideBytes_, rawRowBytes);
  EXPECT_EQ(idxA.metadata().rowStrideBytes_, 64u);
  EXPECT_EQ(idxA.metadata().rowStrideBytes_ % 64u, 0u);
  // The padded `.data` file is correspondingly larger.
  EXPECT_GE(std::filesystem::file_size(aligned.basename + ".vec.al.data"),
            NUM_VECTORS * 64u);

  // Bit-exact top-k parity between the aligned and unaligned stores (the pad
  // tail is never read by the metric).
  for (size_t q = 0; q < NUM_VECTORS; q += 17) {
    auto ru = idxU.searchExact(unaligned.vecs[q], 10);
    auto ra = idxA.searchExact(aligned.vecs[q], 10);
    ASSERT_EQ(ru.size(), ra.size());
    for (size_t i = 0; i < ru.size(); ++i) {
      EXPECT_EQ(ru[i].entity_, ra[i].entity_);
      EXPECT_EQ(ru[i].distance_, ra[i].distance_);  // bit-exact
    }
  }
  cleanupBase(unaligned.basename, unaligned.name);
  cleanupBase(aligned.basename, aligned.name);
}

// _____________________________________________________________________________
// `makeResident(AlignedCopy)` builds a 64-byte-aligned RAM copy of an UNPADDED
// (v4-style) store and serves searches from it with identical results. Also
// exercises `Advise`/`Lock` (best-effort, must never change results).
TEST(VectorIndex, makeResidentAlignedCopyMatches) {
  constexpr size_t dim = 10;  // unpadded rows on disk
  auto b = buildAligned(dim, /*alignRows=*/false, /*withHnsw=*/true, "res");

  // Baseline results from the plain mmap path.
  VectorIndex ref;
  ref.open(b.basename, b.name);
  std::vector<std::vector<ScoredEntity>> baseline;
  for (size_t q = 0; q < NUM_VECTORS; q += 17) {
    baseline.push_back(ref.searchExact(b.vecs[q], 10));
  }

  for (auto residency :
       {VectorIndex::Residency::Advise, VectorIndex::Residency::Lock,
        VectorIndex::Residency::AlignedCopy}) {
    VectorIndex idx;
    idx.open(b.basename, b.name, residency);
    size_t bi = 0;
    for (size_t q = 0; q < NUM_VECTORS; q += 17, ++bi) {
      auto res = idx.searchExact(b.vecs[q], 10);
      ASSERT_EQ(res.size(), baseline[bi].size());
      for (size_t i = 0; i < res.size(); ++i) {
        EXPECT_EQ(res[i].entity_, baseline[bi][i].entity_);
        EXPECT_EQ(res[i].distance_, baseline[bi][i].distance_);
      }
    }
    // The HNSW path must also still find each stored vector's nearest self
    // after the store was possibly repointed at the aligned copy.
    auto hnsw = idx.searchHnswByEntity(b.ids[20], 3);
    ASSERT_FALSE(hnsw.empty());
    EXPECT_EQ(hnsw.front().entity_, b.ids[20]);
  }
  cleanupBase(b.basename, b.name);
}

// _____________________________________________________________________________
// Back-compat: a legacy v4 metadata (no `rowStrideBytes`, version 4) must still
// load, deriving the stride as the raw row byte length, with unchanged results.
TEST(VectorIndex, loadsLegacyV4Metadata) {
  auto b = buildAligned(10, /*alignRows=*/false, /*withHnsw=*/false, "v4");
  VectorIndex ref;
  ref.open(b.basename, b.name);
  auto expected = ref.searchExact(b.vecs[5], 10);

  // Rewrite the metadata to look like a v4 index: version 4 and no
  // `rowStrideBytes` key at all (the field a v4 build never wrote).
  const std::string metaPath = b.basename + ".vec." + b.name + ".meta";
  {
    std::ifstream in{metaPath};
    nlohmann::json j;
    in >> j;
    j["version"] = 4;
    j.erase("rowStrideBytes");
    std::ofstream out{metaPath};
    out << j.dump(2);
  }
  VectorIndex idx;
  idx.open(b.basename, b.name);  // must not throw
  EXPECT_EQ(idx.metadata().version_, 4u);
  EXPECT_EQ(idx.metadata().rowStrideBytes_, 0u);  // absent -> derived at load
  auto got = idx.searchExact(b.vecs[5], 10);
  ASSERT_EQ(got.size(), expected.size());
  for (size_t i = 0; i < got.size(); ++i) {
    EXPECT_EQ(got[i].entity_, expected[i].entity_);
    EXPECT_EQ(got[i].distance_, expected[i].distance_);
  }
  cleanupBase(b.basename, b.name);
}

// _____________________________________________________________________________
// Micro-benchmark (disabled by default): brute-force scan latency for
// {unaligned-cold, unaligned-resident, aligned-resident}. Run with:
//   --gtest_also_run_disabled_tests --gtest_filter='*scanResidencyBenchmark*'
TEST(VectorIndex, DISABLED_scanResidencyBenchmark) {
  const char* nEnv = std::getenv("QLEVER_BENCH_N");
  const size_t n = nEnv ? std::stoull(nEnv) : 200'000;
  constexpr size_t dim = 100;  // 400 raw bytes -> padded to 448
  auto run = [&](bool alignRows, VectorIndex::Residency residency,
                 const char* label) {
    std::string basename = uniqueTmpBasename() + "-bench";
    std::mt19937 rng{7};
    std::normal_distribution<float> g{0.f, 1.f};
    VectorIndexConfig cfg;
    cfg.name_ = "bench";
    cfg.dimensions_ = dim;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.buildHnsw_ = false;
    cfg.alignRows_ = alignRows;
    VectorIndexBuilder builder{basename, cfg};
    std::vector<float> v(dim);
    for (size_t i = 0; i < n; ++i) {
      for (auto& x : v) x = g(rng);
      builder.add(mkId(1000 + i * 3), "<http://ex/" + std::to_string(i) + ">",
                  v);
    }
    builder.build();
    VectorIndex idx;
    idx.open(basename, "bench", residency);
    for (auto& x : v) x = g(rng);
    auto t0 = std::chrono::steady_clock::now();
    constexpr size_t reps = 20;
    for (size_t r = 0; r < reps; ++r) {
      auto res = idx.searchExact(v, 10);
      EXPECT_EQ(res.size(), 10u);
    }
    auto t1 = std::chrono::steady_clock::now();
    double nsPerVec =
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() /
        static_cast<double>(reps * n);
    std::cout << "[scan-bench] " << label << ": " << nsPerVec << " ns/vector"
              << std::endl;
    cleanupBase(basename, "bench");
    return nsPerVec;
  };
  double cold = run(false, VectorIndex::Residency::None, "unaligned-cold");
  double resident =
      run(false, VectorIndex::Residency::Advise, "unaligned-resident");
  double aligned =
      run(true, VectorIndex::Residency::AlignedCopy, "aligned-resident");
  std::cout << "[scan-bench] cold=" << cold << " resident=" << resident
            << " aligned=" << aligned << " ns/vector" << std::endl;
}

#ifdef QLEVER_WITH_PARQUET
#include <arrow/api.h>
#include <arrow/io/file.h>
#include <parquet/arrow/writer.h>

// _____________________________________________________________________________
// The Parquet ingest: a file with `uri` (string) and `embedding`
// (list<float32>) columns round-trips through `ParquetVectorInputReader`.
TEST(VectorIndex, parquetInputReader) {
  std::string path = uniqueTmpBasename() + ".parquet";
  auto* pool = arrow::default_memory_pool();
  arrow::StringBuilder uriBuilder{pool};
  auto valueBuilder = std::make_shared<arrow::FloatBuilder>(pool);
  arrow::ListBuilder listBuilder{pool, valueBuilder};
  constexpr size_t numRows = 10;
  constexpr size_t dim = 4;
  for (size_t i = 0; i < numRows; ++i) {
    // Both bare URIs and bracketed IRIs occur in the wild; the reader passes
    // them through as-is (the build hook normalizes).
    std::string uri = i % 2 == 0 ? "http://ex/" + std::to_string(i)
                                 : "<http://ex/" + std::to_string(i) + ">";
    ASSERT_TRUE(uriBuilder.Append(uri).ok());
    ASSERT_TRUE(listBuilder.Append().ok());
    for (size_t j = 0; j < dim; ++j) {
      ASSERT_TRUE(valueBuilder->Append(static_cast<float>(i * dim + j)).ok());
    }
  }
  std::shared_ptr<arrow::Array> uriArray;
  std::shared_ptr<arrow::Array> embeddingArray;
  ASSERT_TRUE(uriBuilder.Finish(&uriArray).ok());
  ASSERT_TRUE(listBuilder.Finish(&embeddingArray).ok());
  auto schema =
      arrow::schema({arrow::field("uri", arrow::utf8()),
                     arrow::field("embedding", arrow::list(arrow::float32()))});
  auto table = arrow::Table::Make(schema, {uriArray, embeddingArray});
  auto outResult = arrow::io::FileOutputStream::Open(path);
  ASSERT_TRUE(outResult.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*table, pool, outResult.ValueUnsafe(), 1024)
          .ok());

  ParquetVectorInputReader reader{path};
  EXPECT_EQ(reader.numRows(), numRows);
  std::string iri;
  std::vector<float> vec;
  size_t count = 0;
  while (reader.next(iri, vec)) {
    ASSERT_EQ(vec.size(), dim);
    EXPECT_FLOAT_EQ(vec[0], static_cast<float>(count * dim));
    EXPECT_TRUE(iri.find("http://ex/" + std::to_string(count)) !=
                std::string::npos);
    ++count;
  }
  EXPECT_EQ(count, numRows);
  EXPECT_EQ(reader.dimensions(), dim);

  // A file without the expected columns is rejected with a clear error.
  std::string badPath = uniqueTmpBasename() + "-bad.parquet";
  auto badSchema = arrow::schema({arrow::field("uri", arrow::utf8())});
  auto badTable = arrow::Table::Make(badSchema, {uriArray});
  auto badOut = arrow::io::FileOutputStream::Open(badPath);
  ASSERT_TRUE(badOut.ok());
  ASSERT_TRUE(
      parquet::arrow::WriteTable(*badTable, pool, badOut.ValueUnsafe(), 1024)
          .ok());
  EXPECT_THROW(ParquetVectorInputReader{badPath}, std::exception);

  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::filesystem::remove(badPath, ec);
}
#endif  // QLEVER_WITH_PARQUET

// _____________________________________________________________________________
// Manual scale/performance smoke check (excluded from regular runs; it builds
// a multi-million-vector index). Run with:
//   --gtest_also_run_disabled_tests --gtest_filter='*scaleBenchmark*'
TEST(VectorIndex, DISABLED_scaleBenchmark) {
  // Overridable for experiments: QLEVER_BENCH_N / QLEVER_BENCH_THREADS.
  const char* nEnv = std::getenv("QLEVER_BENCH_N");
  const size_t n = nEnv ? std::stoull(nEnv) : 2'000'000;
  const char* threadsEnv = std::getenv("QLEVER_BENCH_THREADS");
  const uint32_t threads =
      threadsEnv ? static_cast<uint32_t>(std::stoul(threadsEnv)) : 0;
  const char* efEnv = std::getenv("QLEVER_BENCH_EF");
  const uint32_t ef = efEnv ? static_cast<uint32_t>(std::stoul(efEnv)) : 64;
  // 0 = iid gaussian -- the known pathological case for ANN (distance
  // concentration; recall then scales with `ef`, not with graph quality).
  // The default is clustered data, which is what real embeddings look like.
  const char* clustersEnv = std::getenv("QLEVER_BENCH_CLUSTERS");
  const size_t clusters = clustersEnv ? std::stoull(clustersEnv) : 1000;
  const char* scalarEnv = std::getenv("QLEVER_BENCH_SCALAR");
  const VectorScalar scalar =
      scalarEnv ? vectorScalarFromString(scalarEnv) : VectorScalar::F32;
  constexpr size_t dim = 128;
  std::string basename = uniqueTmpBasename();
  std::mt19937 rng{42};
  std::normal_distribution<float> g{0.f, 1.f};
  VectorIndexConfig cfg;
  cfg.name_ = "bench";
  cfg.dimensions_ = dim;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = true;
  cfg.buildThreads_ = threads;
  cfg.hnswExpansionSearch_ = ef;
  cfg.scalar_ = scalar;
  std::vector<std::vector<float>> centroids;
  for (size_t c = 0; c < clusters; ++c) {
    std::vector<float> centroid(dim);
    for (auto& x : centroid) x = g(rng);
    centroids.push_back(std::move(centroid));
  }
  auto t0 = std::chrono::steady_clock::now();
  VectorIndexBuilder builder{basename, cfg};
  std::vector<float> v(dim);
  for (size_t i = 0; i < n; ++i) {
    for (auto& x : v) x = g(rng);
    if (clusters > 0) {
      const auto& centroid = centroids[i % clusters];
      for (size_t j = 0; j < dim; ++j) v[j] = centroid[j] + 0.15f * v[j];
    }
    builder.add(mkId(10 + i * 3), "<http://ex/" + std::to_string(i) + ">", v);
  }
  auto t1 = std::chrono::steady_clock::now();
  auto meta = builder.build();
  auto t2 = std::chrono::steady_clock::now();
  ASSERT_EQ(meta.numVectors_, n);
  VectorIndex idx;
  idx.open(basename, "bench");
  // Recall@10 of HNSW vs. exact on a sample of stored vectors.
  size_t hits = 0, total = 0;
  auto t3 = std::chrono::steady_clock::now();
  for (size_t q = 0; q < n; q += n / 50) {
    auto query = idx.getVector(mkId(10 + q * 3));
    ASSERT_TRUE(query.has_value());
    auto exact = idx.searchExact(query.value(), 10);
    auto hnsw = idx.searchHnsw(query.value(), 10);
    std::set<uint64_t> truth;
    for (const auto& e : exact) truth.insert(e.entity_.getBits());
    for (const auto& h : hnsw)
      if (truth.contains(h.entity_.getBits())) ++hits;
    total += exact.size();
  }
  auto t4 = std::chrono::steady_clock::now();
  auto ms = [](auto a, auto b) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
  };
  double recall = static_cast<double>(hits) / static_cast<double>(total);
  std::cout << "[bench] n=" << n << " dim=" << dim << " threads=" << threads
            << " ef=" << ef << " clusters=" << clusters << "\n"
            << "[bench] stream-in: " << ms(t0, t1) << " ms\n"
            << "[bench] build (sort+gather+HNSW): " << ms(t1, t2) << " ms\n"
            << "[bench] 51 exact + 51 hnsw queries: " << ms(t3, t4) << " ms\n"
            << "[bench] recall@10: " << recall << std::endl;
  EXPECT_GE(recall, 0.9);
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.bench" + suffix, ec);
  }
}
