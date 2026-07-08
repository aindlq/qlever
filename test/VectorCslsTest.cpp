// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests of the CSLS (Cross-domain Similarity Local Scaling) retrieval cut:
// the `csls`/`cslsNeighbors`/`cslsR` build keys (the `.csls` r(d) sidecar --
// brute-force, HNSW self-kNN, and verbatim ingestion), the r(d) distribution
// / saturation log, and the query-time `vec:cslsThreshold` filter
// (`vec:bindCsls`, FORM W and FORM P, variable cardinality, cosine stays the
// score).

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "./util/GTestHelpers.h"
#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "global/Id.h"
#include "global/IndexTypes.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "parser/SparqlParser.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/json.h"

namespace {
using ad_utility::testing::getQec;
using ad_utility::testing::makeGetId;
using namespace qlever::vector;
using ::testing::HasSubstr;

constexpr std::string_view PREFIX =
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n"
    "PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>\n";

// The test knowledge graph: five embedded items `<a>`..`<e>`, one item with
// no vector (`<x>`, must be dropped silently by FORM P), and a two-element
// subset `<SubItem>` = {`<b>`, `<c>`} for the FORM P restriction test.
constexpr std::string_view KG_CSLS =
    "<a> <is-a> <CItem> . <b> <is-a> <CItem> . <c> <is-a> <CItem> . "
    "<d> <is-a> <CItem> . <e> <is-a> <CItem> . <x> <is-a> <CItem> . "
    "<b> <is-a> <SubItem> . <c> <is-a> <SubItem> .";

// Unit vectors, dimension 4, cosine metric. Cosine SIMILARITIES:
//   a.b=0.8  a.c=0.6  a.d=0    a.e=0
//   b.c=0.96 b.d=0.6  b.e=0    c.d=0.8  c.e=0  d.e=0
// r(d) with cslsNeighbors=2 (mean of the top-2 sims, SELF-EXCLUDED):
//   a: (0.8+0.6)/2  = 0.7      b: (0.96+0.8)/2 = 0.88
//   c: (0.96+0.8)/2 = 0.88     d: (0.8+0.6)/2  = 0.7      e: 0
// Query q = [0,1,0,0] (== d's stored vector): sims a:0 b:0.6 c:0.8 d:1 e:0.
// r(q) = mean of the top-2 sims EXCLUDING the exact self-match d
//      = (0.8+0.6)/2 = 0.7.
// CSLS = 2*sim - r(q) - r(d):
//   a: -1.4   b: -0.38   c: 0.02   d: 0.6   e: -0.7
// => tau=0 keeps {d, c}; tau=0.1 keeps {d}; tau=-0.5 keeps {d, c, b};
//    tau=-2 keeps all five.
const std::vector<std::pair<std::string, std::vector<float>>>&
cslsTestVectors() {
  static const std::vector<std::pair<std::string, std::vector<float>>> vecs{
      {"<a>", {1.f, 0.f, 0.f, 0.f}},   {"<b>", {0.8f, 0.6f, 0.f, 0.f}},
      {"<c>", {0.6f, 0.8f, 0.f, 0.f}}, {"<d>", {0.f, 1.f, 0.f, 0.f}},
      {"<e>", {0.f, 0.f, 1.f, 0.f}},
  };
  return vecs;
}

// Write `rows` as a NumPy v1.0 `.npy` file (little-endian float32, C-order)
// plus the row-aligned IRI sidecar (the build hook's input bundle).
void writeNpyBundle(
    const std::string& npyPath, const std::string& irisPath,
    const std::vector<std::pair<std::string, std::vector<float>>>& rows) {
  const size_t numRows = rows.size();
  const size_t dim = rows.front().second.size();
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     std::to_string(numRows) + ", " + std::to_string(dim) +
                     "), }";
  size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  std::ofstream out{npyPath, std::ios::binary};
  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t headerLen = static_cast<uint16_t>(dict.size());
  char lenBytes[2] = {static_cast<char>(headerLen & 0xff),
                      static_cast<char>((headerLen >> 8) & 0xff)};
  out.write(lenBytes, 2);
  out.write(dict.data(), dict.size());
  std::ofstream irisOut{irisPath};
  for (const auto& [iri, vec] : rows) {
    out.write(reinterpret_cast<const char*>(vec.data()),
              vec.size() * sizeof(float));
    irisOut << iri << "\n";
  }
}

// Write `values` as a 1-D float32 `.npy` of shape `(N,)` -- the precomputed
// r(d) input of the `cslsR` GPU path.
void writeNpyColumn(const std::string& npyPath,
                    const std::vector<float>& values) {
  std::string dict = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
                     std::to_string(values.size()) + ",), }";
  size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  std::ofstream out{npyPath, std::ios::binary};
  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t headerLen = static_cast<uint16_t>(dict.size());
  char lenBytes[2] = {static_cast<char>(headerLen & 0xff),
                      static_cast<char>((headerLen >> 8) & 0xff)};
  out.write(lenBytes, 2);
  out.write(dict.data(), dict.size());
  out.write(reinterpret_cast<const char*>(values.data()),
            values.size() * sizeof(float));
}

// The shared test context: THREE indices over the same vectors, built through
// the REGISTERED build/load hooks (the `--service-index` path incl. the
// csls keys of `parseSpec`):
//   * "embc"     -- csls-enabled (f32, cslsNeighbors 2, brute-force r(d));
//   * "embr"     -- csls-enabled with a PRECOMPUTED r(d) (`cslsR` ingestion);
//   * "embplain" -- a plain index WITHOUT csls (the validation-error target).
QueryExecutionContext* qecWithCslsIndexes() {
  QueryExecutionContext* qec = getQec(std::string{KG_CSLS});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{VECTOR_EXTENSION_NAME}) != nullptr) {
    return qec;
  }
  std::string basename = (std::filesystem::temp_directory_path() /
                          ("qlever-veccslstest-" + std::to_string(::getpid())))
                             .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  std::string rNpy = basename + ".input.r.npy";
  writeNpyBundle(npy, iris, cslsTestVectors());
  // The known r(d) column of the "embr" ingestion index, row-aligned with the
  // input rows a..e.
  writeNpyColumn(rNpy, {0.1f, 0.2f, 0.3f, 0.4f, 0.5f});
  nlohmann::json spec{
      {"vectorSearch",
       nlohmann::json::array({nlohmann::json{{"name", "embc"},
                                             {"npy", npy},
                                             {"iris", iris},
                                             {"metric", "cosine"},
                                             {"hnsw", false},
                                             {"csls", true},
                                             {"cslsNeighbors", 2}},
                              nlohmann::json{{"name", "embr"},
                                             {"npy", npy},
                                             {"iris", iris},
                                             {"metric", "cosine"},
                                             {"hnsw", false},
                                             {"csls", true},
                                             {"cslsNeighbors", 2},
                                             {"cslsR", rNpy}},
                              nlohmann::json{{"name", "embplain"},
                                             {"npy", npy},
                                             {"iris", iris},
                                             {"metric", "cosine"},
                                             {"hnsw", false}},
                              nlohmann::json{{"name", "embchnsw"},
                                             {"npy", npy},
                                             {"iris", iris},
                                             {"metric", "cosine"},
                                             {"hnsw", true},
                                             {"csls", true},
                                             {"cslsNeighbors", 2}}})}};
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  // The csls builds must produce the sidecar, the plain build must not.
  EXPECT_TRUE(std::filesystem::exists(vectorCslsFile(basename, "embc")));
  EXPECT_TRUE(std::filesystem::exists(vectorCslsFile(basename, "embr")));
  EXPECT_FALSE(std::filesystem::exists(vectorCslsFile(basename, "embplain")));
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  // The load hook memory-maps everything; the directory entries can go.
  for (std::string_view name : {"embc", "embr", "embplain", "embchnsw"}) {
    for (std::string_view suffix :
         {".meta", ".keys", ".rowmap", ".data", ".rerank.data", ".iris",
          ".hnsw", ".csls"}) {
      std::error_code ec;
      std::filesystem::remove(
          basename + ".vec." + std::string{name} + std::string{suffix}, ec);
    }
  }
  for (std::string_view suffix :
       {".input.npy", ".input.iris", ".input.r.npy"}) {
    std::error_code ec;
    std::filesystem::remove(basename + std::string{suffix}, ec);
  }
  return qec;
}

// Parse, plan, and return the execution tree of `query`.
QueryExecutionTree planQuery(QueryExecutionContext* qec, std::string query) {
  ParsedQuery pq = SparqlParser::parseQuery(
      &qec->getIndex().getImpl().encodedIriManager(), std::move(query));
  auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
  QueryPlanner planner{qec, handle};
  return planner.createExecutionTree(pq);
}

// ===========================================================================
// Helpers for the direct-builder (no knowledge graph) structural tests.

// A per-test unique path prefix, so that concurrently running tests (ctest -j)
// never race on shared filenames.
std::string uniqueTmpBasename() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  return (std::filesystem::temp_directory_path() /
          (std::string{"qlever-veccslstest-"} + info->test_suite_name() + "-" +
           info->name()))
      .string();
}

Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

void cleanupTmp(const std::string& basename, const std::string& name) {
  for (std::string_view suffix : {".meta", ".keys", ".rowmap", ".data",
                                  ".rerank.data", ".iris", ".hnsw", ".csls"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec." + name + std::string{suffix},
                            ec);
  }
}

// Deterministic random unit vectors for the exact brute-force r(d) check.
constexpr size_t RD_DIM = 16;
constexpr size_t RD_ROWS = 200;
std::vector<std::vector<float>> makeRandomUnitVectors() {
  std::mt19937 rng{4242};
  std::normal_distribution<float> g{0.f, 1.f};
  std::vector<std::vector<float>> out;
  for (size_t i = 0; i < RD_ROWS; ++i) {
    std::vector<float> v(RD_DIM);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    out.push_back(std::move(v));
  }
  return out;
}

// Hand-computed reference r(d) (double precision): for each row, the mean
// cosine SIMILARITY of its top-`k` nearest OTHER rows (self-excluded).
std::vector<double> referenceRd(const std::vector<std::vector<float>>& vecs,
                                size_t k) {
  const size_t n = vecs.size();
  auto cosSim = [&](size_t a, size_t b) {
    double dot = 0, na = 0, nb = 0;
    for (size_t j = 0; j < vecs[a].size(); ++j) {
      dot += static_cast<double>(vecs[a][j]) * vecs[b][j];
      na += static_cast<double>(vecs[a][j]) * vecs[a][j];
      nb += static_cast<double>(vecs[b][j]) * vecs[b][j];
    }
    return dot / (std::sqrt(na) * std::sqrt(nb));
  };
  std::vector<double> r(n);
  for (size_t i = 0; i < n; ++i) {
    std::vector<double> sims;
    for (size_t j = 0; j < n; ++j) {
      if (j != i) sims.push_back(cosSim(i, j));
    }
    std::sort(sims.rbegin(), sims.rend());
    double sum = 0;
    size_t take = std::min(k, sims.size());
    for (size_t t = 0; t < take; ++t) sum += sims[t];
    r[i] = take == 0 ? 0.0 : sum / static_cast<double>(take);
  }
  return r;
}

std::string buildRandomCslsIndex(bool withHnsw, const std::string& name) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = RD_DIM;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = withHnsw;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 3;
  VectorIndexBuilder builder{basename, cfg};
  auto vecs = makeRandomUnitVectors();
  for (size_t i = 0; i < vecs.size(); ++i) {
    builder.add(mkId(1000 + i * 7), "<http://ex/" + std::to_string(i) + ">",
                vecs[i]);
  }
  builder.build();
  return basename;
}
}  // namespace

// _____________________________________________________________________________
// The csls build writes the `.csls` sidecar (one f32 per row) and records
// `cslsNeighbors` in the `.meta`; the loaded r(d) EXACTLY matches the
// hand-computed brute-force mean-of-top-k-self-cosine (self-excluded) --
// the brute-force fallback runs because the fixture has no HNSW and fewer
// than 50000 vectors.
TEST(VectorCsls, bruteForceRdMatchesHandComputed) {
  std::string basename = buildRandomCslsIndex(/*withHnsw=*/false, "csbf");

  // Sidecar payload: RD_ROWS f32 (plus at most one page of `MmapVector`
  // rounding + its trailer).
  const auto pageSlack = static_cast<std::uintmax_t>(getpagesize()) + 32;
  const std::uintmax_t cslsBytes =
      std::filesystem::file_size(vectorCslsFile(basename, "csbf"));
  EXPECT_GE(cslsBytes, RD_ROWS * sizeof(float));
  EXPECT_LE(cslsBytes, RD_ROWS * sizeof(float) + pageSlack);

  // The `.meta` records the neighbour count.
  {
    std::ifstream in{vectorMetaFile(basename, "csbf")};
    ASSERT_TRUE(in.is_open());
    nlohmann::json j;
    in >> j;
    ASSERT_TRUE(j.contains("cslsNeighbors"));
    EXPECT_EQ(j.at("cslsNeighbors"), 3);
  }

  VectorIndex idx;
  idx.open(basename, "csbf");
  EXPECT_TRUE(idx.hasCsls());
  EXPECT_EQ(idx.cslsNeighbors(), 3u);
  EXPECT_EQ(idx.numVectors(), RD_ROWS);

  auto vecs = makeRandomUnitVectors();
  auto expected = referenceRd(vecs, 3);
  for (size_t i = 0; i < RD_ROWS; ++i) {
    auto r = idx.cslsRForEntity(mkId(1000 + i * 7));
    ASSERT_TRUE(r.has_value()) << i;
    EXPECT_NEAR(r.value(), expected[i], 1e-4) << "row " << i;
  }
  cleanupTmp(basename, "csbf");
}

// _____________________________________________________________________________
// The HNSW self-kNN path (recall-tuned expansion max(200, 20*k)) agrees with
// the exact brute-force reference: at this size the expansion covers the
// whole graph, so recall is essentially perfect.
TEST(VectorCsls, hnswRdMatchesBruteForce) {
  std::string basename = buildRandomCslsIndex(/*withHnsw=*/true, "cshnsw");
  VectorIndex idx;
  idx.open(basename, "cshnsw");
  ASSERT_TRUE(idx.hasCsls());
  ASSERT_TRUE(idx.hasHnsw());

  auto vecs = makeRandomUnitVectors();
  auto expected = referenceRd(vecs, 3);
  size_t tight = 0;
  for (size_t i = 0; i < RD_ROWS; ++i) {
    auto r = idx.cslsRForEntity(mkId(1000 + i * 7));
    ASSERT_TRUE(r.has_value()) << i;
    EXPECT_NEAR(r.value(), expected[i], 5e-3) << "row " << i;
    if (std::abs(r.value() - expected[i]) < 1e-4) {
      ++tight;
    }
  }
  // At least 95% of the rows must match the exact value (an occasional HNSW
  // recall miss on a tie is tolerated).
  EXPECT_GE(tight, (RD_ROWS * 95) / 100);
  cleanupTmp(basename, "cshnsw");
}

// _____________________________________________________________________________
// A csls index defaults to recall-favouring HNSW graph parameters (M 16->32,
// efConstruction 128->256) through `parseSpec`: r(d) is computed once and gates
// every query, so the one-time build spends on recall. A non-csls index keeps
// the usearch stock defaults (16/128).
TEST(VectorCsls, cslsRaisesHnswRecallDefaults) {
  auto* qec = qecWithCslsIndexes();
  auto csls = qlever::vector::getVectorIndex(qec->getIndex(), "embchnsw");
  ASSERT_TRUE(csls != nullptr);
  EXPECT_EQ(csls->metadata().config_.hnswConnectivity_, 32u);
  EXPECT_EQ(csls->metadata().config_.hnswExpansionAdd_, 256u);
  auto plain = qlever::vector::getVectorIndex(qec->getIndex(), "embplain");
  ASSERT_TRUE(plain != nullptr);
  EXPECT_EQ(plain->metadata().config_.hnswConnectivity_, 16u);
  EXPECT_EQ(plain->metadata().config_.hnswExpansionAdd_, 128u);
}

// _____________________________________________________________________________
// Direct-builder `cslsR` ingestion: the given values are stored VERBATIM in
// the `.csls` sidecar, following the rows through the id sort (rows are added
// in descending id order here).
TEST(VectorCsls, cslsRIngestionVerbatim) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "csing";
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 2;
  VectorIndexBuilder builder{basename, cfg};
  // Exact-f32 values, added in DESCENDING entity-id order.
  const std::vector<std::pair<uint64_t, float>> rows{
      {104, 0.75f}, {103, 0.5f}, {102, 0.25f}, {101, 0.125f}};
  for (const auto& [id, r] : rows) {
    builder.add(mkId(id), "<http://ex/" + std::to_string(id) + ">",
                std::vector<float>{1.f, static_cast<float>(id), 0.f, 0.f}, r);
  }
  builder.build();

  VectorIndex idx;
  idx.open(basename, "csing");
  ASSERT_TRUE(idx.hasCsls());
  for (const auto& [id, r] : rows) {
    auto stored = idx.cslsRForEntity(mkId(id));
    ASSERT_TRUE(stored.has_value()) << id;
    EXPECT_EQ(stored.value(), r) << id;  // bit-exact
  }
  cleanupTmp(basename, "csing");
}

// _____________________________________________________________________________
// A mix of rows with and without a cslsR value is rejected (the precomputed
// r(d) must cover every row), as is a cslsR value on a non-csls builder.
TEST(VectorCsls, cslsRIngestionMixRejected) {
  {
    VectorIndexConfig cfg;
    cfg.name_ = "csmix";
    cfg.dimensions_ = 4;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.buildHnsw_ = false;
    cfg.csls_ = true;
    VectorIndexBuilder builder{uniqueTmpBasename(), cfg};
    std::vector<float> v{1.f, 0.f, 0.f, 0.f};
    builder.add(mkId(100), "<http://ex/0>", v, 0.5f);
    AD_EXPECT_THROW_WITH_MESSAGE(builder.add(mkId(101), "<http://ex/1>", v),
                                 HasSubstr("every row"));
  }
  {
    VectorIndexConfig cfg;
    cfg.name_ = "csnone";
    cfg.dimensions_ = 4;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.buildHnsw_ = false;
    VectorIndexBuilder builder{uniqueTmpBasename(), cfg};
    std::vector<float> v{1.f, 0.f, 0.f, 0.f};
    AD_EXPECT_THROW_WITH_MESSAGE(
        builder.add(mkId(100), "<http://ex/0>", v, 0.5f), HasSubstr("csls"));
  }
}

// _____________________________________________________________________________
// Build-time validation: csls is cosine-only, and a binary store without a
// rerank layer (Hamming-only distances) cannot carry it. Without HNSW and
// without a precomputed cslsR, stores of >= 50000 vectors refuse the O(n^2)
// brute-force fallback with a clear error.
TEST(VectorCsls, buildValidationErrors) {
  {
    VectorIndexConfig cfg;
    cfg.name_ = "bad";
    cfg.dimensions_ = 4;
    cfg.metric_ = VectorMetric::L2Sq;
    cfg.csls_ = true;
    AD_EXPECT_THROW_WITH_MESSAGE((VectorIndexBuilder{uniqueTmpBasename(), cfg}),
                                 HasSubstr("cosine"));
  }
  {
    VectorIndexConfig cfg;
    cfg.name_ = "bad";
    cfg.dimensions_ = 8;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.scalar_ = VectorScalar::Binary;
    cfg.csls_ = true;
    AD_EXPECT_THROW_WITH_MESSAGE((VectorIndexBuilder{uniqueTmpBasename(), cfg}),
                                 HasSubstr("Hamming"));
  }
  {
    std::string basename = uniqueTmpBasename();
    VectorIndexConfig cfg;
    cfg.name_ = "big";
    cfg.dimensions_ = 4;
    cfg.metric_ = VectorMetric::Cosine;
    cfg.buildHnsw_ = false;
    cfg.csls_ = true;
    VectorIndexBuilder builder{basename, cfg};
    std::vector<float> v{1.f, 0.f, 0.f, 0.f};
    for (size_t i = 0; i < 50000; ++i) {
      v[1] = static_cast<float>(i % 97);
      builder.add(mkId(100 + i), "<http://ex/" + std::to_string(i) + ">", v);
    }
    AD_EXPECT_THROW_WITH_MESSAGE(builder.build(), HasSubstr("hnsw"));
    cleanupTmp(basename, "big");
  }
}

// _____________________________________________________________________________
// The saturation guard: a fixture of near-identical vectors yields r(d) ~ 1
// everywhere, so the build logs the r(d) distribution AND the near-saturated
// warning (median r(d) >= 0.95 => CSLS ~ 0 => the cut carries no signal).
TEST(VectorCsls, saturationWarning) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "cssat";
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 5;
  VectorIndexBuilder builder{basename, cfg};
  for (size_t i = 0; i < 30; ++i) {
    // Tiny perturbations of one direction: pairwise cosine similarity ~ 1.
    builder.add(
        mkId(100 + i), "<http://ex/" + std::to_string(i) + ">",
        std::vector<float>{1.f, 0.001f * static_cast<float>(i), 0.f, 0.f});
  }
  testing::internal::CaptureStdout();
  builder.build();
  std::string log = testing::internal::GetCapturedStdout();
  EXPECT_THAT(log, HasSubstr("csls r(d): min/p50/p95/max"));
  EXPECT_THAT(log, HasSubstr("near-saturated"));
  cleanupTmp(basename, "cssat");
}

// _____________________________________________________________________________
// A `.meta` without the csls field loads as a plain index: `hasCsls()` is
// false and no sidecar exists (old indexes are unaffected).
TEST(VectorCsls, nonCslsIndexLoadsWithoutCsls) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "plain";
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  VectorIndexBuilder builder{basename, cfg};
  builder.add(mkId(100), "<http://ex/0>", std::vector<float>{1, 0, 0, 0});
  builder.build();
  EXPECT_FALSE(std::filesystem::exists(vectorCslsFile(basename, "plain")));
  {
    std::ifstream in{vectorMetaFile(basename, "plain")};
    nlohmann::json j;
    in >> j;
    EXPECT_FALSE(j.contains("cslsNeighbors"));
  }
  VectorIndex idx;
  idx.open(basename, "plain");
  EXPECT_FALSE(idx.hasCsls());
  cleanupTmp(basename, "plain");
}

// _____________________________________________________________________________
// The extension-level `cslsR` ingestion (the `.npy` GPU path): the given
// per-input-row values are stored verbatim and served back by the loaded
// index.
TEST(VectorCsls, extensionCslsRIngestion) {
  auto* qec = qecWithCslsIndexes();
  auto vidx = getVectorIndex(qec->getIndex(), "embr");
  ASSERT_TRUE(vidx != nullptr);
  ASSERT_TRUE(vidx->hasCsls());
  EXPECT_EQ(vidx->cslsNeighbors(), 2u);
  auto getId = makeGetId(qec->getIndex());
  const std::vector<std::pair<std::string, float>> expected{{"<a>", 0.1f},
                                                            {"<b>", 0.2f},
                                                            {"<c>", 0.3f},
                                                            {"<d>", 0.4f},
                                                            {"<e>", 0.5f}};
  for (const auto& [iri, r] : expected) {
    auto stored = vidx->cslsRForEntity(getId(iri));
    ASSERT_TRUE(stored.has_value()) << iri;
    EXPECT_EQ(stored.value(), r) << iri;  // bit-exact
  }
}

// _____________________________________________________________________________
// The brute-force r(d) of the SERVICE fixture matches the hand-computed
// values of the designed vectors (see `cslsTestVectors`).
TEST(VectorCsls, fixtureRdMatchesHandComputed) {
  auto* qec = qecWithCslsIndexes();
  auto vidx = getVectorIndex(qec->getIndex(), "embc");
  ASSERT_TRUE(vidx != nullptr);
  ASSERT_TRUE(vidx->hasCsls());
  auto getId = makeGetId(qec->getIndex());
  const std::vector<std::pair<std::string, double>> expected{
      {"<a>", 0.7}, {"<b>", 0.88}, {"<c>", 0.88}, {"<d>", 0.7}, {"<e>", 0.0}};
  for (const auto& [iri, r] : expected) {
    auto stored = vidx->cslsRForEntity(getId(iri));
    ASSERT_TRUE(stored.has_value()) << iri;
    EXPECT_NEAR(stored.value(), r, 1e-5) << iri;
  }
}

// _____________________________________________________________________________
// FORM W with `vec:cslsThreshold`: the cut keeps exactly the hand-computed
// survivors, `vec:bindScore` binds the COSINE DISTANCE (not the CSLS value),
// `vec:bindCsls` binds the computed CSLS, results sort by cosine distance
// ascending, and the cardinality is variable: raising tau shrinks the set,
// lowering it grows the set. No `vec:k` => ALL survivors.
TEST(VectorCsls, formWThresholdCut) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());

  auto run = [&](std::string_view tau) {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
                 "vec:bindScore ?d ; vec:bindCsls ?csls ; vec:cslsThreshold " +
                 std::string{tau} + " . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    size_t cslsCol = qet.getVariableColumn(Variable{"?csls"});
    std::vector<std::tuple<Id, double, double>> rows;
    for (size_t r = 0; r < result->idTable().numRows(); ++r) {
      rows.emplace_back(result->idTable()(r, nnCol),
                        result->idTable()(r, dCol).getDouble(),
                        result->idTable()(r, cslsCol).getDouble());
    }
    return rows;
  };

  // tau = 0: survivors {d (csls 0.6), c (csls 0.02)}, ascending by cosine
  // DISTANCE: d (0) before c (0.2).
  auto atZero = run("0.0");
  ASSERT_EQ(atZero.size(), 2u);
  EXPECT_EQ(std::get<0>(atZero[0]), getId("<d>"));
  EXPECT_EQ(std::get<0>(atZero[1]), getId("<c>"));
  // `?d` is the cosine DISTANCE...
  EXPECT_NEAR(std::get<1>(atZero[0]), 0.0, 1e-6);
  EXPECT_NEAR(std::get<1>(atZero[1]), 0.2, 1e-5);
  // ...and `?csls` the CSLS value (they clearly differ).
  EXPECT_NEAR(std::get<2>(atZero[0]), 0.6, 1e-5);
  EXPECT_NEAR(std::get<2>(atZero[1]), 0.02, 1e-5);

  // Raising tau shrinks the set; lowering it grows it (variable cardinality).
  EXPECT_EQ(run("0.1").size(), 1u);
  EXPECT_EQ(std::get<0>(run("0.1")[0]), getId("<d>"));
  auto broader = run("-0.5");
  ASSERT_EQ(broader.size(), 3u);  // {d, c, b}
  EXPECT_EQ(std::get<0>(broader[2]), getId("<b>"));
  EXPECT_NEAR(std::get<2>(broader[2]), -0.38, 1e-5);
  EXPECT_EQ(run("-2.0").size(), 5u);  // everything
}

// _____________________________________________________________________________
// An explicit `vec:k` caps the survivors (top-k by cosine distance); the
// bound scores stay cosine distances.
TEST(VectorCsls, formWThresholdWithKCap) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
               "vec:bindScore ?d ; vec:cslsThreshold -0.5 ; vec:k 1 . } }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  ASSERT_EQ(result->idTable().numRows(), 1u);
  EXPECT_EQ(result->idTable()(0, nnCol), getId("<d>"));
}

// _____________________________________________________________________________
// FORM W with a stored-ENTITY query point (`vec:query <d>`): the query
// entity's own row is the excluded self-match of r(q) (by row identity), so
// the survivors equal the raw-vector form's -- and the entity itself stays a
// RESULT (self-exclusion only affects r(q), not the candidate set).
TEST(VectorCsls, formWEntityQueryPoint) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
               "vec:query <d> ; vec:result ?nn ; vec:bindScore ?d ; "
               "vec:bindCsls ?csls ; vec:cslsThreshold 0.0 . } }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  size_t cslsCol = qet.getVariableColumn(Variable{"?csls"});
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 2u);
  EXPECT_EQ(table(0, nnCol), getId("<d>"));
  EXPECT_EQ(table(1, nnCol), getId("<c>"));
  EXPECT_NEAR(table(0, cslsCol).getDouble(), 0.6, 1e-5);
  EXPECT_NEAR(table(1, cslsCol).getDouble(), 0.02, 1e-5);
}

// _____________________________________________________________________________
// FORM P (bound candidates): the csls cut runs over EXACTLY the bound set --
// r(q) comes from the candidates, non-members are dropped, and the search
// never pulls in non-candidates. With the subset {b, c} and tau = 0, only c
// survives (r(q) over {b, c} = 0.7; csls(b) = -0.38, csls(c) = 0.02) --
// although the whole-index cut would also keep d.
TEST(VectorCsls, formPRestrictedToBoundSet) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { ?x <is-a> <SubItem> . "
               "SERVICE vec: { _:c vec:index \"embc\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:candidates ?x ; "
               "vec:result ?x ; vec:bindScore ?d ; vec:bindCsls ?csls ; "
               "vec:cslsThreshold 0.0 . } }");
  auto result = qet.getResult();
  size_t xCol = qet.getVariableColumn(Variable{"?x"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t cslsCol = qet.getVariableColumn(Variable{"?csls"});
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 1u);
  EXPECT_EQ(table(0, xCol), getId("<c>"));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.2, 1e-5);      // cosine distance
  EXPECT_NEAR(table(0, cslsCol).getDouble(), 0.02, 1e-5);  // CSLS value
}

// _____________________________________________________________________________
// FORM P annotate over the full member set: candidates without a stored
// vector (`<x>`) are dropped silently, all tau-survivors are kept (no
// `vec:k`), and the bound values match the hand-computed FORM W ones (the
// bound set == the member set here).
TEST(VectorCsls, formPAnnotateAllSurvivors) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { ?e <is-a> <CItem> . "
               "SERVICE vec: { _:c vec:index \"embc\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:candidates ?e ; "
               "vec:result ?e ; vec:bindScore ?d ; vec:bindCsls ?csls ; "
               "vec:cslsThreshold -0.5 . } }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t cslsCol = qet.getVariableColumn(Variable{"?csls"});
  const IdTable& table = result->idTable();
  // Survivors at tau = -0.5: {b, c, d}; `<x>` (no vector) and the cut ones
  // are gone.
  ASSERT_EQ(table.numRows(), 3u);
  ad_utility::HashMap<Id, double> cslsByEntity;
  for (size_t r = 0; r < table.numRows(); ++r) {
    cslsByEntity[table(r, eCol)] = table(r, cslsCol).getDouble();
  }
  ASSERT_TRUE(cslsByEntity.contains(getId("<b>")));
  ASSERT_TRUE(cslsByEntity.contains(getId("<c>")));
  ASSERT_TRUE(cslsByEntity.contains(getId("<d>")));
  EXPECT_FALSE(cslsByEntity.contains(getId("<x>")));
  EXPECT_NEAR(cslsByEntity[getId("<b>")], -0.38, 1e-5);
  EXPECT_NEAR(cslsByEntity[getId("<c>")], 0.02, 1e-5);
  EXPECT_NEAR(cslsByEntity[getId("<d>")], 0.6, 1e-5);
}

// _____________________________________________________________________________
// `vec:cslsThreshold` on an index that was NOT built with csls fails with a
// clear runtime error (FORM W and FORM P).
TEST(VectorCsls, thresholdOnNonCslsIndexErrors) {
  auto* qec = qecWithCslsIndexes();
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { SERVICE vec: { _:c vec:index "
                           "\"embplain\" ; vec:queryVector \"0,1,0,0\" ; "
                           "vec:result ?nn ; vec:cslsThreshold 0.0 . } }");
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(),
                                 HasSubstr("was not built with csls:true"));
  }
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { ?e <is-a> <CItem> . "
                           "SERVICE vec: { _:c vec:index \"embplain\" ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:candidates ?e ; "
                           "vec:result ?e ; vec:cslsThreshold 0.0 . } }");
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(),
                                 HasSubstr("was not built with csls:true"));
  }
}

// _____________________________________________________________________________
// Build-spec validation through the registered build hook: csls requires the
// cosine metric, and the csls sub-keys require `csls: true`.
TEST(VectorCsls, buildSpecValidation) {
  auto* qec = qecWithCslsIndexes();
  ASSERT_FALSE(IndexExtensionRegistry::get().buildHooks().empty());
  const auto& hook = IndexExtensionRegistry::get().buildHooks().front();
  AD_EXPECT_THROW_WITH_MESSAGE(
      hook(qec->getIndex(), "/nonexistent/base",
           nlohmann::json::parse(
               R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
               R"("iris":"/nonexistent.iris","metric":"l2sq",)"
               R"("csls":true}]})")),
      HasSubstr("cosine"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      hook(qec->getIndex(), "/nonexistent/base",
           nlohmann::json::parse(
               R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
               R"("iris":"/nonexistent.iris","cslsNeighbors":5}]})")),
      HasSubstr("csls"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      hook(qec->getIndex(), "/nonexistent/base",
           nlohmann::json::parse(
               R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
               R"("iris":"/nonexistent.iris","scalar":"binary",)"
               R"("csls":true}]})")),
      HasSubstr("Hamming"));
}

// _____________________________________________________________________________
// Parse-time validation of the new SERVICE parameters.
TEST(VectorCsls, parseErrors) {
  auto* qec = qecWithCslsIndexes();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
  // cslsThreshold requires a query point (FORM E has none).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:candidates ?c ; "
                           "vec:result ?x ; vec:cslsThreshold 0.0 .")),
      HasSubstr("requires a query point"));
  // bindCsls / cslsNeighbors require cslsThreshold.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:bindCsls ?c .")),
      HasSubstr("requires `<cslsThreshold>`"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; "
                           "vec:cslsNeighbors 5 .")),
      HasSubstr("requires `<cslsThreshold>`"));
  // The csls cut is a full fine scan: no coarse pass, no HNSW.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:bindCoarseScore ?dc .")),
      HasSubstr("bindCoarseScore"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:rerankK 100 .")),
      HasSubstr("rerankK"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:algorithm vec:hnsw .")),
      HasSubstr("hnsw"));
  // Non-positive cslsNeighbors; non-numeric threshold; shared variables.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:cslsNeighbors 0 .")),
      HasSubstr("positive integer"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; "
                           "vec:cslsThreshold \"high\" .")),
      HasSubstr("finite number"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:bindCsls ?x .")),
      HasSubstr("must be different"));
}
