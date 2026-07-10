// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests of the CSLS (Cross-domain Similarity Local Scaling) retrieval cut:
// the `csls`/`cslsNeighbors`/`cslsR` build keys (the `.csls` r(d) sidecar --
// exact brute force on the FINE layer, the dedicated fine-layer HNSW
// self-kNN, and verbatim ingestion; always independent of the main query
// graph and of `hnsw:`), the r(d) distribution / saturation log, and the
// query-time `vec:cslsThreshold` filter (`vec:bindCsls`, FORM W and FORM P,
// variable cardinality, cosine stays the score).

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numbers>
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
#include "services/vectorSearch/VectorSearch.h"
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

// Write `rows` as a bf16 `.npy` (descr '<V2', the ml_dtypes bfloat16 on-disk
// form: each value is the top 16 bits of its fp32, little-endian) + the IRIs.
// Same bundle as `writeNpyBundle` but the exact input format the user builds
// from -- the path none of the f32 tests exercise.
void writeBf16NpyBundle(
    const std::string& npyPath, const std::string& irisPath,
    const std::vector<std::pair<std::string, std::vector<float>>>& rows) {
  const size_t numRows = rows.size();
  const size_t dim = rows.front().second.size();
  std::string dict = "{'descr': '<V2', 'fortran_order': False, 'shape': (" +
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
    for (float v : vec) {
      uint32_t bits = 0;
      std::memcpy(&bits, &v, sizeof(bits));
      auto bf16 = static_cast<uint16_t>(bits >> 16);  // truncate to bf16
      char b[2] = {static_cast<char>(bf16 & 0xff),
                   static_cast<char>((bf16 >> 8) & 0xff)};
      out.write(b, 2);
    }
    irisOut << iri << "\n";
  }
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
  std::string bnpy = basename + ".input.bf16.npy";
  std::string bIris = basename + ".input.bf16.iris";
  writeNpyBundle(npy, iris, cslsTestVectors());
  // Same vectors as `embc`, but fed as a bf16 `.npy` -- the exact input format
  // the user builds from (the f32 path is otherwise the only one tested).
  writeBf16NpyBundle(bnpy, bIris, cslsTestVectors());
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
                                             {"cslsNeighbors", 2}},
                              nlohmann::json{{"name", "embbf16"},
                                             {"npy", bnpy},
                                             {"iris", bIris},
                                             {"metric", "cosine"},
                                             {"scalar", "binary"},
                                             {"rerank", "bf16"},
                                             {"hnsw", false},
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
  for (std::string_view name :
       {"embc", "embr", "embplain", "embchnsw", "embbf16"}) {
    for (std::string_view suffix :
         {".meta", ".keys", ".rowmap", ".data", ".rerank.data", ".iris",
          ".hnsw", ".csls"}) {
      std::error_code ec;
      std::filesystem::remove(
          basename + ".vec." + std::string{name} + std::string{suffix}, ec);
    }
  }
  for (std::string_view suffix : {".input.npy", ".input.iris", ".input.r.npy",
                                  ".input.bf16.npy", ".input.bf16.iris"}) {
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

std::string buildRandomCslsIndex(
    bool withHnsw, const std::string& name,
    VectorScalar scalar = VectorScalar::F32,
    std::optional<VectorScalar> rerank = std::nullopt,
    std::optional<size_t> bruteForceMaxForTesting = std::nullopt) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = RD_DIM;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = withHnsw;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 3;
  cfg.scalar_ = scalar;
  cfg.rerankScalar_ = rerank;
  VectorIndexBuilder builder{basename, cfg};
  if (bruteForceMaxForTesting.has_value()) {
    builder.setCslsBruteForceMaxForTesting(bruteForceMaxForTesting.value());
  }
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
// hand-computed brute-force mean-of-top-k-self-cosine (self-excluded). The
// exact brute force runs because the store is below the brute-force bound --
// and it needs NO main graph at all: `hnsw: false` builds the sidecar and no
// `.hnsw` file (csls is decoupled from the query-time ANN).
TEST(VectorCsls, bruteForceRdMatchesHandComputed) {
  std::string basename = buildRandomCslsIndex(/*withHnsw=*/false, "csbf");
  EXPECT_FALSE(std::filesystem::exists(vectorHnswFile(basename, "csbf")));

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
// The DEDICATED fine-layer HNSW self-kNN path (stores at or above the
// brute-force bound; forced here by lowering the bound below RD_ROWS) agrees
// with the exact brute-force reference -- WITHOUT any main graph
// (`hnsw: false`, no `.hnsw` file): csls builds, self-searches, and discards
// its own graph. At this size the recall-tuned expansion covers the whole
// graph, so recall is essentially perfect.
TEST(VectorCsls, dedicatedFineGraphRdMatchesBruteForce) {
  std::string basename = buildRandomCslsIndex(
      /*withHnsw=*/false, "csdedic", VectorScalar::F32, std::nullopt,
      /*bruteForceMaxForTesting=*/10);
  // Decoupling: the dedicated csls graph is never persisted, and no main
  // graph was requested.
  EXPECT_FALSE(std::filesystem::exists(vectorHnswFile(basename, "csdedic")));
  VectorIndex idx;
  idx.open(basename, "csdedic");
  ASSERT_TRUE(idx.hasCsls());
  ASSERT_FALSE(idx.hasHnsw());

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
  cleanupTmp(basename, "csdedic");
}

// _____________________________________________________________________________
// The `hnsw*` keys and their defaults concern the MAIN query-time graph only:
// a csls index keeps the usearch stock defaults (16/128) exactly like a plain
// index. (The recall-favouring parameters -- M 32, efConstruction 256, plus
// the high self-search expansion -- now live on the DEDICATED fine-layer
// graph the builder creates and discards for the r(d) self-kNN; they are not
// a property of the persisted index.)
TEST(VectorCsls, cslsLeavesMainGraphDefaultsUntouched) {
  auto* qec = qecWithCslsIndexes();
  auto csls = qlever::vector::getVectorIndex(qec->getIndex(), "embchnsw");
  ASSERT_TRUE(csls != nullptr);
  EXPECT_EQ(csls->metadata().config_.hnswConnectivity_, 16u);
  EXPECT_EQ(csls->metadata().config_.hnswExpansionAdd_, 128u);
  auto plain = qlever::vector::getVectorIndex(qec->getIndex(), "embplain");
  ASSERT_TRUE(plain != nullptr);
  EXPECT_EQ(plain->metadata().config_.hnswConnectivity_, 16u);
  EXPECT_EQ(plain->metadata().config_.hnswExpansionAdd_, 128u);
}

// _____________________________________________________________________________
// r(d) from a BF16 `.npy` input (binary+bf16 -- the user's exact build format)
// must match the f32-input r(d) for the SAME vectors. A saturated (~1.0) result
// here would pin the bug on the bf16 input path, not the data.
TEST(VectorCsls, bf16NpyInputRdMatchesF32Input) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  auto f32 = qlever::vector::getVectorIndex(qec->getIndex(), "embc");
  auto bf16 = qlever::vector::getVectorIndex(qec->getIndex(), "embbf16");
  ASSERT_TRUE(f32 != nullptr);
  ASSERT_TRUE(bf16 != nullptr);
  for (std::string_view iri : {"<a>", "<b>", "<c>", "<d>", "<e>"}) {
    Id e = getId(std::string{iri});
    auto rf = f32->cslsRForEntity(e);
    auto rb = bf16->cslsRForEntity(e);
    ASSERT_TRUE(rf.has_value()) << iri;
    ASSERT_TRUE(rb.has_value()) << iri;
    EXPECT_NEAR(rb.value(), rf.value(), 3e-2f)
        << iri << ": bf16 r(d)=" << rb.value() << " vs f32 r(d)=" << rf.value();
  }
}

// _____________________________________________________________________________
// The ON-DISK csls r(d) sidecar the RUNTIME uses (read via the loaded index)
// exposes a real r(d) spread -- min < max, matching the known fixture values --
// NOT a degenerate all-1.0. I.e. what is persisted is what the build computed;
// the load-time diagnostic (`loaded csls r(d) sidecar: min/p50/p95/max = ...`,
// full-precision, in VectorIndex::open) reports these same values on startup.
TEST(VectorCsls, loadedCslsSidecarIsRealDistribution) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  auto vidx = qlever::vector::getVectorIndex(qec->getIndex(), "embc");
  ASSERT_TRUE(vidx != nullptr);
  std::vector<float> r;
  for (std::string_view iri : {"<a>", "<b>", "<c>", "<d>", "<e>"}) {
    auto v = vidx->cslsRForEntity(getId(std::string{iri}));
    ASSERT_TRUE(v.has_value()) << iri;
    r.push_back(v.value());
  }
  std::sort(r.begin(), r.end());
  // Fixture r(d) (cslsNeighbors=2): a=0.7 b=0.88 c=0.88 d=0.7 e=0.
  EXPECT_NEAR(r.front(), 0.0f, 1e-3f);
  EXPECT_NEAR(r.back(), 0.88f, 1e-2f);
  EXPECT_LT(r.front(), r.back())
      << "a degenerate all-equal (1/1/1) sidecar would fail here";
}

// _____________________________________________________________________________
// Binary scan + bf16 rerank WITHOUT a main graph (`hnsw: false`): the r(d)
// self-kNN runs entirely on the FINE bf16 (cosine) layer -- the coarse binary
// Hamming bytes never participate -- so csls builds a correct `.csls` (and no
// `.hnsw`) on a two-layer index with no query-time ANN at all. r(d) is a real
// cosine similarity tracking the exact reference, NOT a Hamming-derived score
// (which would collapse toward saturation). Regression for a coarse-layer
// leak into CSLS on a low-precision index.
TEST(VectorCsls, binaryScanRdComputedOnBf16WithoutMainGraph) {
  std::string basename = buildRandomCslsIndex(
      /*withHnsw=*/false, "csbin", VectorScalar::Binary, VectorScalar::Bf16);
  EXPECT_FALSE(std::filesystem::exists(vectorHnswFile(basename, "csbin")));
  EXPECT_TRUE(std::filesystem::exists(vectorCslsFile(basename, "csbin")));
  VectorIndex idx;
  idx.open(basename, "csbin");
  ASSERT_TRUE(idx.hasCsls());
  ASSERT_FALSE(idx.hasHnsw());
  auto expected = referenceRd(makeRandomUnitVectors(), 3);
  size_t tight = 0;
  size_t saturated = 0;
  for (size_t i = 0; i < RD_ROWS; ++i) {
    auto r = idx.cslsRForEntity(mkId(1000 + i * 7));
    ASSERT_TRUE(r.has_value());
    if (std::abs(r.value() - static_cast<float>(expected[i])) < 2e-2f) {
      ++tight;  // bf16 truncation tolerance
    }
    if (r.value() > 0.95f) {
      ++saturated;
    }
  }
  // Tracks the exact bf16-cosine reference for the vast majority (the exact
  // brute force runs on the bf16 layer, so the only error is bf16 precision),
  // and is NOT saturated -- a collapse to ~1.0 would be a coarse-layer leak.
  EXPECT_GE(tight, (RD_ROWS * 85) / 100);
  EXPECT_LT(saturated, RD_ROWS / 4) << "r(d) saturated -- coarse-layer leak?";
  cleanupTmp(basename, "csbin");
}

// _____________________________________________________________________________
// THE anisotropic regression (reproduce-then-fix): real embedding spaces are
// anisotropic -- every vector sits near a common direction, so the 1-bit sign
// codes of a `binary` scan layer nearly COLLIDE (here: all components of the
// common direction are +0.125, the per-component noise sd is 0.06, so < 5% of
// all sign bits flip and the Hamming neighbourhoods are near-degenerate ties)
// while the bf16 cosines stay GRADED (nearest-other well below 1). The old
// build computed the r(d) self-kNN through the MAIN Hamming graph, whose
// "nearest neighbours" on such data are not the bf16-nearest at all -- r(d)
// came out wrong (saturated on real corpora), which is impossible: r(d) is a
// MEAN of the top-k neighbour cosines, so it can never exceed the true top-k
// mean. The fix computes r(d) via a DEDICATED bf16-cosine HNSW over the FINE
// layer (forced here by lowering the brute-force bound below N), so it must
// match the exact brute-force fine reference and stay un-saturated.
TEST(VectorCsls, anisotropicBinaryRdMatchesFineReferenceNotSaturated) {
  constexpr size_t N = 1500;
  constexpr size_t D = 64;
  constexpr size_t K = 8;
  std::mt19937 rng{20260708};
  std::normal_distribution<float> noise{0.f, 0.06f};
  // All vectors = common unit direction (1,...,1)/8 + small noise.
  std::vector<std::vector<float>> vecs(N, std::vector<float>(D));
  size_t negativeComponents = 0;
  for (auto& v : vecs) {
    for (auto& x : v) {
      x = 0.125f + noise(rng);
      if (x <= 0.f) ++negativeComponents;
    }
  }
  // Fixture guard 1: the sign codes nearly collide (almost all bits are 1).
  EXPECT_LT(negativeComponents, (N * D) / 20);
  // Fixture guard 2: the fine cosines are GRADED -- the exact reference r(d)
  // is itself well below saturation everywhere.
  auto expected = referenceRd(vecs, K);
  EXPECT_LT(*std::max_element(expected.begin(), expected.end()), 0.95);

  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "csaniso";
  cfg.dimensions_ = D;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;  // csls needs NO main graph anymore
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = K;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  VectorIndexBuilder builder{basename, cfg};
  // Force the dedicated fine-layer HNSW path (N >= the bound).
  builder.setCslsBruteForceMaxForTesting(100);
  for (size_t i = 0; i < N; ++i) {
    builder.add(mkId(1000 + i * 7), "<http://ex/" + std::to_string(i) + ">",
                vecs[i]);
  }
  builder.build();
  EXPECT_FALSE(std::filesystem::exists(vectorHnswFile(basename, "csaniso")));

  VectorIndex idx;
  idx.open(basename, "csaniso");
  ASSERT_TRUE(idx.hasCsls());
  size_t tight = 0;
  size_t saturated = 0;
  double sumAbsErr = 0;
  for (size_t i = 0; i < N; ++i) {
    auto r = idx.cslsRForEntity(mkId(1000 + i * 7));
    ASSERT_TRUE(r.has_value()) << i;
    const double err = r.value() - expected[i];
    // r(d) is a mean of top-K neighbour cosines: it can NEVER exceed the
    // exact reference (+ bf16 truncation slack). The old Hamming-graph path
    // violated this on anisotropic data.
    EXPECT_LE(err, 2e-2) << "row " << i;
    // ... and an HNSW recall miss may only lose a little.
    EXPECT_GE(err, -5e-2) << "row " << i;
    sumAbsErr += std::abs(err);
    if (std::abs(err) < 1e-2) ++tight;
    if (r.value() > 0.95f) ++saturated;
  }
  EXPECT_GE(tight, (N * 90) / 100);
  EXPECT_LT(sumAbsErr / N, 5e-3);
  EXPECT_EQ(saturated, 0u)
      << saturated << "/" << N
      << " rows r(d) > 0.95 -- the coarse Hamming layer leaked into r(d)?";
  cleanupTmp(basename, "csaniso");
}

// _____________________________________________________________________________
// The dedicated fine self-kNN AT SCALE on isotropic data: N (1200) far above
// the (lowered) brute-force bound, binary+bf16, no main graph. r(d) must stay
// SPREAD and track the exact reference -- random unit vectors in dim-48 have
// moderate neighbour cosines (~0.4-0.6), so a wall of ~1.0 would be a
// coarse-path bug, not the data.
TEST(VectorCsls, binaryScanRdAtScaleTracksReference) {
  constexpr size_t N = 1200;
  constexpr size_t D = 48;
  std::mt19937 rng{777};
  std::normal_distribution<float> g{0.f, 1.f};
  std::vector<std::vector<float>> vecs(N, std::vector<float>(D));
  for (auto& v : vecs) {
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
  }
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "csbinbig";
  cfg.dimensions_ = D;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 10;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  VectorIndexBuilder builder{basename, cfg};
  builder.setCslsBruteForceMaxForTesting(100);
  for (size_t i = 0; i < N; ++i) {
    builder.add(mkId(1000 + i * 7), "<http://ex/" + std::to_string(i) + ">",
                vecs[i]);
  }
  builder.build();
  VectorIndex idx;
  idx.open(basename, "csbinbig");
  auto expected = referenceRd(vecs, 10);
  size_t tight = 0;
  size_t saturated = 0;
  for (size_t i = 0; i < N; ++i) {
    auto r = idx.cslsRForEntity(mkId(1000 + i * 7));
    ASSERT_TRUE(r.has_value());
    if (std::abs(r.value() - static_cast<float>(expected[i])) < 2e-2f) {
      ++tight;  // bf16 truncation + rare HNSW-miss tolerance
    }
    if (r.value() > 0.95f) {
      ++saturated;
    }
  }
  EXPECT_GE(tight, (N * 90) / 100);
  EXPECT_LT(saturated, N / 10)
      << saturated << "/" << N << " rows r(d) > 0.95 (coarse-path leak?)";
  cleanupTmp(basename, "csbinbig");
}

// _____________________________________________________________________________
// TWO-LAYER (binary+bf16) query-time csls cut: the coarse-scan + bounded
// fine-rerank path must return EXACTLY the survivors of a full fine sweep --
// same entities in the same order, same (fine) cosine scores, same csls
// values. The full-sweep reference is the same index with a huge
// `cslsRerankFloor` (one batch reranks ALL candidates == the old full-sweep
// path); a pruning floor (400 of 1200) and a small floor (64 -- both
// thresholds keep far more than 64 survivors, so a single batch cannot cover
// them and the WIDEN loop must extend the reranked set) must both reproduce
// it. Two thresholds, raw-vector and entity query points.
//
// The fixture is TIERED around a common center (150 near vectors at 10-20
// degrees, 100 mid vectors at 45-55 degrees, 2250 random background): the cut
// regions of the two thresholds (~150 and ~250 survivors, picked from the
// fixture's own csls distribution) are then COMPACT in coarse (Hamming) rank,
// which is the regime the widen loop is built for -- with dim-256 sign codes
// the tiers' Hamming bands (~14-28, ~64-78, ~128) are cleanly separated, so
// batches beyond the cut region come up empty and stop the widening only
// after every survivor was reranked. (Fully random low-dimensional data has
// no such coarse-fine correlation, and no floor short of "everything" -- the
// production default relative to these sizes -- could guarantee equality.)
//
// N deliberately exceeds `VEC_SEARCH_PARALLEL_THRESHOLD` (2048) so the coarse
// pass runs its PARALLEL histogram-select + parallel collect (binary scan =
// integer Hamming), whose result this asserts is bit-identical to the
// full-fine-sweep reference.
TEST(VectorCsls, twoLayerCoarseRerankMatchesFullFineSweep) {
  constexpr size_t N = 2500;
  constexpr size_t NEAR = 150;
  constexpr size_t MID = 100;
  constexpr size_t D = 256;
  constexpr size_t K = 10;
  std::mt19937 rng{20260709};
  std::normal_distribution<float> g{0.f, 1.f};
  auto randomUnit = [&] {
    std::vector<float> v(D);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
  };
  const std::vector<float> center = randomUnit();
  // A unit vector at an angle uniform in [degLo, degHi] from `center`, in a
  // fresh random plane through it.
  auto tiered = [&](float degLo, float degHi) {
    std::vector<float> w = randomUnit();
    float dot = 0;
    for (size_t j = 0; j < D; ++j) dot += w[j] * center[j];
    float norm = 0;
    for (size_t j = 0; j < D; ++j) {
      w[j] -= dot * center[j];
      norm += w[j] * w[j];
    }
    norm = std::sqrt(norm);
    const float a = std::uniform_real_distribution<float>{degLo, degHi}(
                        rng)*std::numbers::pi_v<float> /
                    180.f;
    std::vector<float> v(D);
    for (size_t j = 0; j < D; ++j) {
      v[j] = std::cos(a) * center[j] + std::sin(a) * w[j] / norm;
    }
    return v;
  };
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "cs2layer";
  cfg.dimensions_ = D;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = K;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  VectorIndexBuilder builder{basename, cfg};
  for (size_t i = 0; i < N; ++i) {
    std::vector<float> v = i < NEAR         ? tiered(10.f, 20.f)
                           : i < NEAR + MID ? tiered(45.f, 55.f)
                                            : randomUnit();
    builder.add(mkId(1000 + i * 7), "<http://ex/" + std::to_string(i) + ">", v);
  }
  builder.build();
  VectorIndex idx;
  idx.open(basename, "cs2layer");
  ASSERT_TRUE(idx.hasCsls());
  ASSERT_TRUE(idx.hasRerankLayer());
  EXPECT_EQ(idx.cslsRerankFloor(), DEFAULT_CSLS_RERANK_FLOOR);

  auto expectSameHits = [](const std::vector<CslsScoredEntity>& expected,
                           const std::vector<CslsScoredEntity>& actual) {
    ASSERT_EQ(expected.size(), actual.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_EQ(expected[i].entity_, actual[i].entity_) << "hit " << i;
      EXPECT_EQ(expected[i].distance_, actual[i].distance_) << "hit " << i;
      EXPECT_EQ(expected[i].csls_, actual[i].csls_) << "hit " << i;
    }
  };
  // Two thresholds picked from the fixture's own csls distribution of a
  // keep-all reference run (csls >= -4 always, so tau = -10 keeps every
  // candidate): one at the near/mid tier boundary (~150 survivors), one at
  // the mid/background boundary (~250) -- both far more than the small floor
  // of 64, so those runs can only be correct if they widen.
  auto pickTaus = [](const std::vector<CslsScoredEntity>& all) {
    std::vector<float> csls;
    csls.reserve(all.size());
    for (const auto& hit : all) csls.push_back(hit.csls_);
    std::sort(csls.rbegin(), csls.rend());
    return std::pair{(csls[NEAR - 1] + csls[NEAR]) / 2.f,
                     (csls[NEAR + MID - 1] + csls[NEAR + MID]) / 2.f};
  };

  const std::vector<float>& query = center;
  idx.setCslsRerankFloor(1'000'000);
  auto all = idx.searchCsls(query, /*threshold=*/-10.f, K);
  ASSERT_EQ(all.size(), N);
  const auto [tauNear, tauMid] = pickTaus(all);
  for (float tau : {tauNear, tauMid}) {
    // Full-fine-sweep reference: the huge floor reranks ALL candidates.
    idx.setCslsRerankFloor(1'000'000);
    size_t numScoredRef = 0;
    auto ref = idx.searchCsls(query, tau, K, std::nullopt, std::nullopt, {},
                              &numScoredRef);
    EXPECT_EQ(numScoredRef, N);
    ASSERT_LT(ref.size(), N);
    // Proof that the small floor below cannot cover the survivors in one
    // batch (so the widen loop is genuinely exercised).
    ASSERT_GT(ref.size(), 64u);
    // A pruning floor: only the coarse-best 400 of 1200 get a fine distance.
    // `numScored` must still report the whole matched set.
    idx.setCslsRerankFloor(400);
    size_t numScoredPruned = 0;
    auto pruned = idx.searchCsls(query, tau, K, std::nullopt, std::nullopt, {},
                                 &numScoredPruned);
    EXPECT_EQ(numScoredPruned, N);
    expectSameHits(ref, pruned);
    // A small floor: the widen loop must keep extending the reranked set
    // until every survivor is found.
    idx.setCslsRerankFloor(64);
    auto widened = idx.searchCsls(query, tau, K);
    expectSameHits(ref, widened);
  }

  // The entity-query form (per-layer STORED row bytes as the query points,
  // the self row excluded from r(q)) must agree the same way.
  const Id queryEntity = mkId(1000 + 5 * 7);  // a near-tier member
  idx.setCslsRerankFloor(1'000'000);
  auto allEnt = idx.searchCslsByEntity(queryEntity, /*threshold=*/-10.f, K);
  ASSERT_EQ(allEnt.size(), N);
  const auto [tauEntNear, tauEntMid] = pickTaus(allEnt);
  for (float tau : {tauEntNear, tauEntMid}) {
    idx.setCslsRerankFloor(1'000'000);
    auto ref = idx.searchCslsByEntity(queryEntity, tau, K);
    ASSERT_GT(ref.size(), 64u);
    ASSERT_LT(ref.size(), N);
    idx.setCslsRerankFloor(64);
    auto widened = idx.searchCslsByEntity(queryEntity, tau, K);
    expectSameHits(ref, widened);
  }
  cleanupTmp(basename, "cs2layer");
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
// rerank layer (Hamming-only distances) cannot carry it. There is NO
// hnsw-related restriction: csls never needs the main graph (above the
// brute-force bound it builds its own fine-layer self-kNN graph), so
// `hnsw: false` is always fine -- covered by the build tests above.
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
// The SOFTMAX autoCut is csls-independent (top-N cosine standouts, no r(d)), so
// -- unlike `vec:cslsThreshold` and `vec:autoCut "csls"` -- it is ACCEPTED on a
// plain (non-csls) cosine index ("embplain"): the query executes to a real
// table with no `.csls` sidecar. The CSLS-knee on the same index still errors.
TEST(VectorCsls, softmaxAutoCutAcceptedOnNonCslsIndex) {
  auto* qec = qecWithCslsIndexes();
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { SERVICE vec: { _:c vec:index "
                           "\"embplain\" ; vec:queryVector \"0,1,0,0\" ; "
                           "vec:result ?nn ; vec:autoCut \"softmax\" . } }");
    // Must NOT throw the csls-availability error -- it runs to a real table.
    auto result = qet.getResult();
    EXPECT_LE(result->idTable().numRows(), 5u);
  }
  {
    // The CSLS-knee DOES read r(d), so it is still rejected on the same index.
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { SERVICE vec: { _:c vec:index "
                           "\"embplain\" ; vec:queryVector \"0,1,0,0\" ; "
                           "vec:result ?nn ; vec:autoCut \"csls\" . } }");
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
  // The csls cut manages its own scan/rerank: no top-k coarse-pass
  // parameters, no HNSW.
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

// ===========================================================================
// The DYNAMIC `vec:autoCut` cuts (knee + softmax) on top of the CSLS
// machinery.

namespace {
// Build a SINGLE-LAYER f32 csls index whose rows are unit vectors in the
// (e0, e1) plane at the given ANGLES (degrees) from e0, with an INGESTED
// r(d) = 0 for every row: `csls = 2*cos(angle) - r(q)`, so the csls ORDER is
// exactly the cosine order and every csls gap is 2x the cosine gap --
// hand-designable knee/softmax fixtures. Row i gets entity id `1000 + i`.
std::string buildAngleFixtureIndex(const std::string& name,
                                   const std::vector<float>& angleDegrees,
                                   size_t cslsNeighbors = 3, bool csls = true,
                                   bool ingestRd = true) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = csls;
  cfg.cslsNeighbors_ = cslsNeighbors;
  VectorIndexBuilder builder{basename, cfg};
  // Ingest a neutral r(d)=0 for the csls fixtures (the cut tests isolate the
  // cut logic from r(d) variation); a plain index takes no r(d) at all. Pass
  // `ingestRd=false` to instead COMPUTE r(d) via the self-kNN (the path that
  // also calibrates the softmax T).
  const std::optional<float> rd =
      (csls && ingestRd) ? std::optional<float>{0.f} : std::nullopt;
  for (size_t i = 0; i < angleDegrees.size(); ++i) {
    const float a = angleDegrees[i] * std::numbers::pi_v<float> / 180.f;
    builder.add(mkId(1000 + i), "<http://ex/" + std::to_string(i) + ">",
                std::vector<float>{std::cos(a), std::sin(a), 0.f, 0.f}, rd);
  }
  builder.build();
  return basename;
}

const std::vector<float> kAngleQuery{1.f, 0.f, 0.f, 0.f};

CslsCut kneeCut(float floor) {
  CslsCut cut;
  cut.mode_ = CslsCut::Mode::Knee;
  cut.threshold_ = floor;
  return cut;
}

CslsCut softmaxCut(size_t n) {
  CslsCut cut;
  cut.mode_ = CslsCut::Mode::Softmax;
  cut.softmaxN_ = n;
  return cut;
}
}  // namespace

// _____________________________________________________________________________
// KNEE on a fixture with a clear cluster + gap: 6 vectors at 10..15 degrees
// (csls ~0.95..0.99 with the ingested r(d) = 0) and 41 background vectors at
// 60..80 degrees (csls <= ~0.02). The cluster->background csls gap (~0.93) is
// far above 3x the median gap (~0.015), so the knee fires at rank 5 and keeps
// EXACTLY the cluster -- although the generous floor (-2) would keep
// everything.
TEST(VectorCsls, autoCutKneeKeepsExactlyTheCluster) {
  std::vector<float> angles{10.f, 11.f, 12.f, 13.f, 14.f, 15.f};
  for (float a = 60.f; a <= 80.f; a += 0.5f) {
    angles.push_back(a);
  }
  std::string basename = buildAngleFixtureIndex("acknee", angles);
  VectorIndex idx;
  idx.open(basename, "acknee");
  ASSERT_TRUE(idx.hasCsls());

  size_t numScored = 0;
  auto hits = idx.searchCsls(kAngleQuery, kneeCut(-2.f), 3, std::nullopt,
                             std::nullopt, {}, &numScored);
  EXPECT_EQ(numScored, angles.size());
  ASSERT_EQ(hits.size(), 6u);
  for (size_t i = 0; i < 6; ++i) {
    // Ascending by cosine distance = ascending by angle = fixture order.
    EXPECT_EQ(hits[i].entity_, mkId(1000 + i)) << i;
    const double cosA = std::cos(angles[i] * std::numbers::pi / 180.0);
    EXPECT_NEAR(hits[i].distance_, 1.0 - cosA, 1e-4) << i;
    // csls = 2*cos - r(q), r(q) = mean of the top-3 cosines (no exact
    // self-match: the nearest candidate is 10 degrees away).
    const double rq = (std::cos(10 * std::numbers::pi / 180.0) +
                       std::cos(11 * std::numbers::pi / 180.0) +
                       std::cos(12 * std::numbers::pi / 180.0)) /
                      3.0;
    EXPECT_NEAR(hits[i].csls_, 2.0 * cosA - rq, 1e-4) << i;
  }
  cleanupTmp(basename, "acknee");
}

// _____________________________________________________________________________
// KNEE significance fallback on a SMOOTH fixture (50 vectors evenly spread
// over 20..80 degrees, no cluster): the largest gap is < 3x the median gap,
// so the knee never fires and the cut degrades to the fixed floor -- with a
// generous floor EVERYTHING is kept, and with a mid-range floor the result is
// IDENTICAL to the fixed-threshold cut at that floor (entities, distances,
// and csls values).
TEST(VectorCsls, autoCutKneeFallsBackWithoutSignificantKnee) {
  std::vector<float> angles;
  for (size_t i = 0; i < 50; ++i) {
    angles.push_back(20.f + static_cast<float>(i) * 60.f / 49.f);
  }
  std::string basename = buildAngleFixtureIndex("acsmooth", angles);
  VectorIndex idx;
  idx.open(basename, "acsmooth");

  // Generous floor: the fallback keeps everything >= floor, i.e. all 50.
  auto all = idx.searchCsls(kAngleQuery, kneeCut(-2.f), 3);
  EXPECT_EQ(all.size(), 50u);

  // Mid-range floor: identical to the fixed cut at the same value.
  const float floor = 0.35f;
  auto knee = idx.searchCsls(kAngleQuery, kneeCut(floor), 3);
  auto fixed = idx.searchCsls(kAngleQuery, CslsCut{floor}, 3);
  ASSERT_EQ(knee.size(), fixed.size());
  ASSERT_LT(knee.size(), 50u);  // the floor itself does cut
  ASSERT_GT(knee.size(), 2u);
  for (size_t i = 0; i < knee.size(); ++i) {
    EXPECT_EQ(knee[i].entity_, fixed[i].entity_) << i;
    EXPECT_EQ(knee[i].distance_, fixed[i].distance_) << i;
    EXPECT_EQ(knee[i].csls_, fixed[i].csls_) << i;
  }
  cleanupTmp(basename, "acsmooth");
}

// _____________________________________________________________________________
// SOFTMAX on a clear-winner fixture: one vector at 5 degrees, 40 background
// vectors at 60..80 degrees. In the top-15 softmax at T = 0.1 the winner
// holds essentially all the mass (p ~ 0.98 >> the standout bar 2/15), the
// best background candidate has p ~ 0.007 -- exactly the winner is kept, with
// its COSINE distance as the score and NO csls value (NaN).
TEST(VectorCsls, autoCutSoftmaxKeepsTheStandout) {
  std::vector<float> angles{5.f};
  for (float a = 60.f; a < 80.f; a += 0.5f) {
    angles.push_back(a);
  }
  std::string basename = buildAngleFixtureIndex("acsoft", angles);
  VectorIndex idx;
  idx.open(basename, "acsoft");

  auto hits = idx.searchCsls(kAngleQuery, softmaxCut(15), 3);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].entity_, mkId(1000));
  EXPECT_NEAR(hits[0].distance_, 1.0 - std::cos(5 * std::numbers::pi / 180.0),
              1e-4);
  EXPECT_TRUE(std::isnan(hits[0].csls_));
  cleanupTmp(basename, "acsoft");
}

// _____________________________________________________________________________
// SOFTMAX no-match rejection on a NEAR-UNIFORM fixture: 40 identical vectors
// at 60 degrees. Every p_i is exactly 1/15 < the standout bar 2/15, so
// NOTHING is kept -- the confidence cut "shrugs" instead of returning an
// arbitrary top-k.
TEST(VectorCsls, autoCutSoftmaxKeepsNothingOnUniform) {
  std::vector<float> angles(40, 60.f);
  std::string basename = buildAngleFixtureIndex("acuni", angles);
  VectorIndex idx;
  idx.open(basename, "acuni");

  size_t numScored = 0;
  auto hits = idx.searchCsls(kAngleQuery, softmaxCut(15), 3, std::nullopt,
                             std::nullopt, {}, &numScored);
  EXPECT_EQ(numScored, 40u);
  EXPECT_TRUE(hits.empty());
  cleanupTmp(basename, "acuni");
}

// _____________________________________________________________________________
// The SOFTMAX cut is CSLS-INDEPENDENT: it thresholds the softmax of the top-N
// COSINE similarities and never reads r(d)/r(q). So it must run on a PLAIN
// cosine index that was NOT built with `csls:true` (no `.csls` sidecar), and
// return the very same standout as the csls-built fixture above.
TEST(VectorCsls, autoCutSoftmaxRunsOnPlainCosineIndex) {
  std::vector<float> angles{5.f};
  for (float a = 60.f; a < 80.f; a += 0.5f) {
    angles.push_back(a);
  }
  std::string basename = buildAngleFixtureIndex("acsoftplain", angles,
                                                /*cslsNeighbors=*/3,
                                                /*csls=*/false);
  VectorIndex idx;
  idx.open(basename, "acsoftplain");
  ASSERT_FALSE(idx.hasCsls());

  auto hits = idx.searchCsls(kAngleQuery, softmaxCut(15), 3);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].entity_, mkId(1000));
  EXPECT_NEAR(hits[0].distance_, 1.0 - std::cos(5 * std::numbers::pi / 180.0),
              1e-4);
  EXPECT_TRUE(std::isnan(hits[0].csls_));
  cleanupTmp(basename, "acsoftplain");
}

// _____________________________________________________________________________
// BUILD-TIME softmax-T calibration: the self-kNN that computes r(d) also
// records each point's neighbour-cosine spread; the median (clamped) becomes
// the softmax serving-default T, persisted in the `.meta` and re-loaded. A
// tight angular cluster (near-identical neighbours) calibrates a small/floored
// T; a spread corpus a larger one. `resolveCslsCut` uses it below any query or
// runtime override.
TEST(VectorCsls, softmaxTemperatureCalibratedFromCorpusSpread) {
  std::vector<float> tightAngles;
  for (float a = 40.f; a < 50.f; a += 0.5f) {
    tightAngles.push_back(a);
  }
  const std::vector<float> spreadAngles{0.f, 20.f, 40.f, 60.f, 80.f};
  // csls with COMPUTED r(d) (ingestRd=false) -> the calibration path runs.
  std::string tightBase =
      buildAngleFixtureIndex("caltight", tightAngles, /*cslsNeighbors=*/3,
                             /*csls=*/true, /*ingestRd=*/false);
  std::string spreadBase =
      buildAngleFixtureIndex("calspread", spreadAngles, /*cslsNeighbors=*/3,
                             /*csls=*/true, /*ingestRd=*/false);
  VectorIndex tight, spread;
  tight.open(tightBase, "caltight");
  spread.open(spreadBase, "calspread");

  ASSERT_TRUE(tight.calibratedSoftmaxTemperature().has_value());
  ASSERT_TRUE(spread.calibratedSoftmaxTemperature().has_value());
  const float tT = tight.calibratedSoftmaxTemperature().value();
  const float sT = spread.calibratedSoftmaxTemperature().value();
  // Clamped into the sane range, and the tighter corpus gives the sharper T.
  EXPECT_GE(tT, 0.01f);
  EXPECT_LE(sT, 0.5f);
  EXPECT_LT(tT, sT);

  // `resolveCslsCut` uses the calibrated T as the softmax default...
  using CoverageMode = VectorSearchConfiguration::CoverageMode;
  using CutSignal = VectorSearchConfiguration::CutSignal;
  VectorSearchConfiguration config;
  config.indexName_ = "calspread";
  config.queryVector_ = kAngleQuery;
  config.autoCut_ = CoverageMode::Balanced;
  config.cutSignal_ = CutSignal::Softmax;
  EXPECT_FLOAT_EQ(resolveCslsCut(config, spread).temperature_, sT);
  // ...but a per-query override still wins.
  config.softmaxTemperature_ = 0.25f;
  EXPECT_FLOAT_EQ(resolveCslsCut(config, spread).temperature_, 0.25f);

  cleanupTmp(tightBase, "caltight");
  cleanupTmp(spreadBase, "calspread");
}

// _____________________________________________________________________________
// INGESTED r(d) (the `cslsR` GPU path) skips the self-kNN, so there is no
// neighbour spread to measure: no calibrated T is persisted, and the query
// path falls back to the constant softmax default.
TEST(VectorCsls, ingestedRdLeavesSoftmaxTUncalibrated) {
  const std::vector<float> angles{0.f, 20.f, 40.f, 60.f, 80.f};
  std::string base = buildAngleFixtureIndex("calingest", angles);  // ingestRd
  VectorIndex idx;
  idx.open(base, "calingest");
  EXPECT_FALSE(idx.calibratedSoftmaxTemperature().has_value());
  cleanupTmp(base, "calingest");
}

// _____________________________________________________________________________
// TWO-LAYER (binary + bf16) autoCut: the softmax cut is bounded by ONE rerank
// batch -- it must find the standout WITHOUT ever widening (no "csls rerank
// widened" log even though a fixed cut at a generous tau would widen through
// everything) -- and the knee cut reproduces the huge-rerank-floor reference
// exactly under a small floor (the widen loop covers every survivor, so the
// knee input -- and output -- is identical).
TEST(VectorCsls, autoCutTwoLayerBoundedSoftmaxAndKneeIdentity) {
  constexpr size_t N_BG = 150;
  constexpr size_t D = 64;
  std::mt19937 rng{20260709};
  std::normal_distribution<float> g{0.f, 1.f};
  auto randomUnit = [&] {
    std::vector<float> v(D);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
  };
  const std::vector<float> query = randomUnit();
  // The standout: 5 degrees from the query in a fresh random plane.
  std::vector<float> standout;
  {
    std::vector<float> w = randomUnit();
    float dot = 0;
    for (size_t j = 0; j < D; ++j) dot += w[j] * query[j];
    float norm = 0;
    for (size_t j = 0; j < D; ++j) {
      w[j] -= dot * query[j];
      norm += w[j] * w[j];
    }
    norm = std::sqrt(norm);
    const float a = 5.f * std::numbers::pi_v<float> / 180.f;
    standout.resize(D);
    for (size_t j = 0; j < D; ++j) {
      standout[j] = std::cos(a) * query[j] + std::sin(a) * w[j] / norm;
    }
  }
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "ac2layer";
  cfg.dimensions_ = D;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = true;
  cfg.cslsNeighbors_ = 3;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  VectorIndexBuilder builder{basename, cfg};
  builder.add(mkId(500), "<http://ex/standout>", standout);
  for (size_t i = 0; i < N_BG; ++i) {
    builder.add(mkId(1000 + i), "<http://ex/" + std::to_string(i) + ">",
                randomUnit());
  }
  builder.build();
  VectorIndex idx;
  idx.open(basename, "ac2layer");
  ASSERT_TRUE(idx.hasRerankLayer());

  // Softmax under a SMALL rerank floor: one batch of 32 (of 151) suffices --
  // the standout is coarse-rank ~1 -- and the widen loop must NOT fire.
  idx.setCslsRerankFloor(32);
  testing::internal::CaptureStdout();
  auto soft = idx.searchCsls(query, softmaxCut(15), 3);
  std::string log = testing::internal::GetCapturedStdout();
  ASSERT_EQ(soft.size(), 1u);
  EXPECT_EQ(soft[0].entity_, mkId(500));
  EXPECT_TRUE(std::isnan(soft[0].csls_));
  EXPECT_THAT(log, HasSubstr("csls softmax: top-15"));
  EXPECT_THAT(log, ::testing::Not(HasSubstr("csls rerank widened")));

  // Knee identity: a generous floor (-10) keeps every candidate in play, so
  // the small-floor run widens through everything and must equal the
  // huge-floor (single batch reranks all) reference bit for bit.
  idx.setCslsRerankFloor(1'000'000);
  auto ref = idx.searchCsls(query, kneeCut(-10.f), 3);
  idx.setCslsRerankFloor(32);
  auto bounded = idx.searchCsls(query, kneeCut(-10.f), 3);
  ASSERT_EQ(ref.size(), bounded.size());
  ASSERT_GE(ref.size(), 1u);
  EXPECT_EQ(ref[0].entity_, mkId(500));
  for (size_t i = 0; i < ref.size(); ++i) {
    EXPECT_EQ(ref[i].entity_, bounded[i].entity_) << i;
    EXPECT_EQ(ref[i].distance_, bounded[i].distance_) << i;
    EXPECT_EQ(ref[i].csls_, bounded[i].csls_) << i;
  }
  cleanupTmp(basename, "ac2layer");
}

// _____________________________________________________________________________
// `resolveCslsCut`: the fixed threshold passthrough, and the coverage-mode ->
// (z-threshold / softmax-alpha / no-match) mapping across the cosine, csls,
// and softmax signals, plus the query-param -> per-index-default -> constant
// chain of the softmax knobs.
TEST(VectorCsls, resolveCutCoverageModeMappingAndDefaults) {
  std::string basename =
      buildAngleFixtureIndex("acres", {0.f, 30.f, 60.f, 90.f},
                             /*cslsNeighbors=*/2);
  VectorIndex idx;
  idx.open(basename, "acres");

  VectorSearchConfiguration config;
  config.indexName_ = "acres";
  config.queryVector_ = kAngleQuery;

  // Fixed threshold: a passthrough.
  config.cslsThreshold_ = 0.75f;
  auto fixed = resolveCslsCut(config, idx);
  EXPECT_EQ(fixed.mode_, CslsCut::Mode::Threshold);
  EXPECT_EQ(fixed.threshold_, 0.75f);
  config.cslsThreshold_ = std::nullopt;

  using CoverageMode = VectorSearchConfiguration::CoverageMode;
  using CutSignal = VectorSearchConfiguration::CutSignal;

  // Cosine signal (the default): a top-anchored ZCut whose band width is the
  // mode's Delta, keeping (except in exact) at least the single best.
  config.cutSignal_ = CutSignal::Cosine;
  struct Case {
    CoverageMode mode_;
    float delta_;
    bool keepOne_;
  };
  for (const Case& c :
       {Case{CoverageMode::Precise, DEFAULT_ZCUT_DELTA_PRECISE, true},
        Case{CoverageMode::Balanced, DEFAULT_ZCUT_DELTA_BALANCED, true},
        Case{CoverageMode::Broad, DEFAULT_ZCUT_DELTA_BROAD, true},
        Case{CoverageMode::Exact, DEFAULT_ZCUT_DELTA_PRECISE, false}}) {
    config.autoCut_ = c.mode_;
    auto cut = resolveCslsCut(config, idx);
    EXPECT_EQ(cut.mode_, CslsCut::Mode::ZCut);
    EXPECT_EQ(cut.signal_, CslsCut::Signal::Cosine);
    EXPECT_FLOAT_EQ(cut.delta_, c.delta_);
    EXPECT_EQ(cut.keepAtLeastOne_, c.keepOne_);
    // The widen depth uses the broadest band + a margin (mode-independent).
    EXPECT_FLOAT_EQ(cut.widenDelta_,
                    DEFAULT_ZCUT_DELTA_BROAD + DEFAULT_ZCUT_WIDEN_MARGIN);
  }

  // Csls signal: the same band widths, but the csls signal.
  config.cutSignal_ = CutSignal::Csls;
  config.autoCut_ = CoverageMode::Broad;
  auto csls = resolveCslsCut(config, idx);
  EXPECT_EQ(csls.mode_, CslsCut::Mode::ZCut);
  EXPECT_EQ(csls.signal_, CslsCut::Signal::Csls);
  EXPECT_FLOAT_EQ(csls.delta_, DEFAULT_ZCUT_DELTA_BROAD);

  // Softmax signal: alpha per mode, N = 5 * cslsNeighbors (2 here), the
  // calibrated/default T.
  config.cutSignal_ = CutSignal::Softmax;
  config.autoCut_ = CoverageMode::Balanced;
  auto soft = resolveCslsCut(config, idx);
  EXPECT_EQ(soft.mode_, CslsCut::Mode::Softmax);
  EXPECT_FLOAT_EQ(soft.temperature_, DEFAULT_CSLS_SOFTMAX_TEMPERATURE);
  EXPECT_EQ(soft.softmaxN_, 10u);
  EXPECT_FLOAT_EQ(soft.alpha_, DEFAULT_ZCUT_SOFTMAX_ALPHA_BALANCED);
  EXPECT_TRUE(soft.keepAtLeastOne_);
  config.autoCut_ = CoverageMode::Precise;
  EXPECT_FLOAT_EQ(resolveCslsCut(config, idx).alpha_,
                  DEFAULT_ZCUT_SOFTMAX_ALPHA_PRECISE);
  config.autoCut_ = CoverageMode::Broad;
  EXPECT_FLOAT_EQ(resolveCslsCut(config, idx).alpha_,
                  DEFAULT_ZCUT_SOFTMAX_ALPHA_BROAD);
  config.autoCut_ = CoverageMode::Exact;
  EXPECT_FALSE(resolveCslsCut(config, idx).keepAtLeastOne_);
  config.autoCut_ = CoverageMode::Balanced;
  // The query-side `cslsNeighbors` override scales the default softmaxN.
  config.cslsNeighbors_ = 4;
  EXPECT_EQ(resolveCslsCut(config, idx).softmaxN_, 20u);
  config.cslsNeighbors_ = std::nullopt;

  // Per-index softmax serving defaults apply when the query does not override,
  // and the query params always win over them.
  idx.setSoftmaxTemperatureDefault(0.2f);
  idx.setSoftmaxNDefault(7);
  auto served = resolveCslsCut(config, idx);
  EXPECT_FLOAT_EQ(served.temperature_, 0.2f);
  EXPECT_EQ(served.softmaxN_, 7u);
  config.softmaxTemperature_ = 0.05f;
  config.softmaxN_ = 3;
  auto overridden = resolveCslsCut(config, idx);
  EXPECT_FLOAT_EQ(overridden.temperature_, 0.05f);
  EXPECT_EQ(overridden.softmaxN_, 3u);

  // Per-index z-cut serving defaults (the `zcut*` env keys) thread into the
  // resolved band width, gate, floor fraction, and widen depth.
  config.cutSignal_ = CutSignal::Cosine;
  config.autoCut_ = CoverageMode::Precise;
  idx.setZcutDeltaDefault(0, 0.5f);
  idx.setZcutDeltaDefault(2, 4.f);
  idx.setZcutGateZDefault(5.f);
  idx.setZcutFloorFractionDefault(0.25f);
  auto zserved = resolveCslsCut(config, idx);
  EXPECT_FLOAT_EQ(zserved.delta_, 0.5f);
  EXPECT_FLOAT_EQ(zserved.gateZ_, 5.f);
  EXPECT_FLOAT_EQ(zserved.floorFraction_, 0.25f);
  // The widen depth follows the (overridden) BROADEST band.
  EXPECT_FLOAT_EQ(zserved.widenDelta_, 4.f + DEFAULT_ZCUT_WIDEN_MARGIN);
  // An unset band width stays at the constant.
  config.autoCut_ = CoverageMode::Balanced;
  EXPECT_FLOAT_EQ(resolveCslsCut(config, idx).delta_,
                  DEFAULT_ZCUT_DELTA_BALANCED);

  cleanupTmp(basename, "acres");
}

// _____________________________________________________________________________
// The SERVICE surface of the dynamic coverage cuts on the hand-computed
// fixture (r(q) = 0.7 for q = [0,1,0,0]; csls: d 0.6, c 0.02, b -0.38, e -0.7,
// a -1.4; d is the clear standout).
TEST(VectorCsls, autoCutServiceForms) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  // Back-compat `vec:autoCut "csls"` = the CSLS-signal z-cut (balanced): the
  // standout d is kept, its top row bound with the CSLS value; `vec:bindCsls`
  // works.
  {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
                 "vec:bindScore ?d ; vec:bindCsls ?csls ; "
                 "vec:autoCut \"csls\" . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    size_t cslsCol = qet.getVariableColumn(Variable{"?csls"});
    const IdTable& table = result->idTable();
    ASSERT_GE(table.numRows(), 1u);
    EXPECT_EQ(table(0, nnCol), getId("<d>"));
    EXPECT_NEAR(table(0, cslsCol).getDouble(), 0.6, 1e-5);
  }
  // Back-compat `vec:autoCut "softmax"` = the softmax signal (balanced): keeps
  // the standout {d}.
  {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
                 "vec:bindScore ?d ; vec:autoCut \"softmax\" . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 1u);
    EXPECT_EQ(table(0, nnCol), getId("<d>"));
    EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-6);
  }
  // The default cosine signal (no `vec:cutSignal`) needs no csls sidecar: it
  // runs on a plain cosine index and keeps the standout d.
  {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embplain\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
                 "vec:autoCut \"precise\" . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    const IdTable& table = result->idTable();
    ASSERT_GE(table.numRows(), 1u);
    EXPECT_EQ(table(0, nnCol), getId("<d>"));
  }
  // FORM P `exact`: the softmax cut over the BOUND subset {b, c} keeps nothing
  // (no standout), and `exact` reports that as the empty result (the
  // non-exact modes would keep the single best instead).
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { ?x <is-a> <SubItem> . "
                           "SERVICE vec: { _:c vec:index \"embc\" ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:candidates ?x ; "
                           "vec:result ?x ; vec:cutSignal \"softmax\" ; "
                           "vec:autoCut \"exact\" . } }");
    auto result = qet.getResult();
    EXPECT_EQ(result->idTable().numRows(), 0u);
  }
  // The runtime csls-availability check fires for the `csls` signal on a
  // non-csls index.
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT * WHERE { SERVICE vec: { _:c vec:index "
                           "\"embplain\" ; vec:queryVector \"0,1,0,0\" ; "
                           "vec:result ?nn ; vec:autoCut \"csls\" . } }");
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(),
                                 HasSubstr("was not built with csls:true"));
  }
}

// _____________________________________________________________________________
// Parse-time validation of the `vec:autoCut` parameter family.
TEST(VectorCsls, autoCutParseErrors) {
  auto* qec = qecWithCslsIndexes();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
  // cslsThreshold and autoCut are mutually exclusive.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:index \"embc\" ; vec:result ?x ; "
                      "vec:queryVector \"0,1,0,0\" ; vec:cslsThreshold 0.0 ; "
                      "vec:autoCut \"csls\" .")),
      HasSubstr("mutually exclusive"));
  // Only the coverage words (+ back-compat "csls"/"softmax") are valid autoCut
  // values.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; "
                           "vec:autoCut \"knee\" .")),
      HasSubstr("\"precise\""));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut 1 .")),
      HasSubstr("string literal"));
  // Only the three signals are valid cutSignal values.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:cutSignal \"knee\" .")),
      HasSubstr("\"cosine\""));
  // cutSignal requires autoCut.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; "
                           "vec:cutSignal \"cosine\" .")),
      HasSubstr("requires `<autoCut>`"));
  // A back-compat `vec:autoCut "softmax"` conflicting with an explicit
  // `vec:cutSignal` is rejected.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"softmax\" ; vec:cutSignal \"cosine\" .")),
      HasSubstr("Conflicting"));
  // autoCut requires a query point (FORM E has none).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:candidates ?c ; "
                           "vec:result ?x ; vec:autoCut \"balanced\" .")),
      HasSubstr("requires a query point"));
  // The softmax knobs require the softmax signal.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:softmaxTemperature 0.2 .")),
      HasSubstr("requires `<cutSignal> \"softmax\"`"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:softmaxN 5 .")),
      HasSubstr("requires `<cutSignal> \"softmax\"`"));
  // bindCsls: rejected with the cosine/softmax signals (no CSLS value).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:bindCsls ?c .")),
      HasSubstr("requires `<cslsThreshold>` or"));
  // Value validation of the knobs.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(
          qec,
          query("_:c vec:index \"embc\" ; vec:result ?x ; "
                "vec:queryVector \"0,1,0,0\" ; vec:cutSignal \"softmax\" ; "
                "vec:autoCut \"balanced\" ; vec:softmaxTemperature 0.0 .")),
      HasSubstr("positive finite"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(
          qec,
          query("_:c vec:index \"embc\" ; vec:result ?x ; "
                "vec:queryVector \"0,1,0,0\" ; vec:cutSignal \"softmax\" ; "
                "vec:autoCut \"balanced\" ; vec:softmaxN 0 .")),
      HasSubstr("positive integer"));
  // The existing csls-cut incompatibilities apply to autoCut too.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:rerankK 100 .")),
      HasSubstr("rerankK"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:algorithm vec:hnsw .")),
      HasSubstr("hnsw"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:autoCut "
                           "\"balanced\" ; vec:bindCoarseScore ?dc .")),
      HasSubstr("bindCoarseScore"));
  // The coverage presets + cutSignal parse (a smoke check).
  planQuery(qec, query("_:c vec:index \"embc\" ; vec:result ?x ; "
                       "vec:queryVector \"0,1,0,0\" ; vec:autoCut \"broad\" ; "
                       "vec:cutSignal \"csls\" ."));
}

// _____________________________________________________________________________
// The noise-floor Z-CUT (`vec:cutSignal "cosine"/"csls"`). `CslsReranked` is a
// public struct, so these hand-build a reranked set to test the cut math
// directly (independent of any index scan), then apply it through the public
// `VectorIndex::applyCslsCut`.
namespace {
using CoverageMode = VectorSearchConfiguration::CoverageMode;

qlever::vector::CslsReranked makeReranked(const std::vector<float>& cosSims,
                                          const std::vector<uint64_t>& ids) {
  qlever::vector::CslsReranked r;
  r.scored_ = cosSims.size();
  r.plateauFound_ = true;
  for (size_t j = 0; j < cosSims.size(); ++j) {
    r.entityBits_.push_back(mkId(ids[j]).getBits());
    r.storeRows_.push_back(0);  // unused by the cosine signal
    r.fineDist_.push_back(1.f - cosSims[j]);
  }
  return r;
}

qlever::vector::CslsCut cosineZCut(CoverageMode mode) {
  qlever::vector::CslsCut cut;
  cut.mode_ = qlever::vector::CslsCut::Mode::ZCut;
  cut.signal_ = qlever::vector::CslsCut::Signal::Cosine;
  cut.gateZ_ = DEFAULT_ZCUT_GATE_Z;
  switch (mode) {
    case CoverageMode::Precise:
      cut.delta_ = DEFAULT_ZCUT_DELTA_PRECISE;
      cut.keepAtLeastOne_ = true;
      break;
    case CoverageMode::Broad:
      cut.delta_ = DEFAULT_ZCUT_DELTA_BROAD;
      cut.keepAtLeastOne_ = true;
      break;
    case CoverageMode::Exact:
      cut.delta_ = DEFAULT_ZCUT_DELTA_PRECISE;
      cut.keepAtLeastOne_ = false;
      break;
    case CoverageMode::Balanced:
    default:
      cut.delta_ = DEFAULT_ZCUT_DELTA_BALANCED;
      cut.keepAtLeastOne_ = true;
      break;
  }
  return cut;
}

std::set<uint64_t> keptIds(
    const std::vector<qlever::vector::CslsScoredEntity>& hits) {
  std::set<uint64_t> s;
  for (const auto& h : hits) s.insert(h.entity_.getBits());
  return s;
}
}  // namespace

// A clear standout is kept by every non-degenerate mode, and -- the cross-modal
// fix -- SCALING every cosine by a constant leaves the cut unchanged (z-scores
// are scale-invariant).
TEST(VectorCsls, zCutStandoutAndScaleInvariance) {
  std::string base = buildAngleFixtureIndex("zstand", {0.f, 30.f, 60.f, 90.f},
                                            /*cslsNeighbors=*/3,
                                            /*csls=*/false);
  VectorIndex idx;
  idx.open(base, "zstand");
  const std::vector<uint64_t> ids{10, 11, 12, 13, 14};
  // One standout at 0.9, a flat background near 0.45.
  auto r = makeReranked({0.9f, 0.5f, 0.48f, 0.46f, 0.44f}, ids);
  for (CoverageMode m : {CoverageMode::Precise, CoverageMode::Balanced,
                         CoverageMode::Broad, CoverageMode::Exact}) {
    auto hits = idx.applyCslsCut(r, cosineZCut(m));
    ASSERT_GE(hits.size(), 1u);
    EXPECT_EQ(hits[0].entity_, mkId(10));  // the standout is always the top
  }
  // Scale ALL cosines by 0.1: the same entities survive (proving the
  // cross-modal scale invariance).
  auto rScaled = makeReranked({0.09f, 0.05f, 0.048f, 0.046f, 0.044f}, ids);
  for (CoverageMode m :
       {CoverageMode::Precise, CoverageMode::Balanced, CoverageMode::Broad}) {
    EXPECT_EQ(keptIds(idx.applyCslsCut(r, cosineZCut(m))),
              keptIds(idx.applyCslsCut(rScaled, cosineZCut(m))))
        << "scale invariance, mode " << static_cast<int>(m);
  }
  cleanupTmp(base, "zstand");
}

// A graded head (dense steps near the best) above a background: a wider band
// (larger Delta) keeps strictly more, and precise's keep set is a subset of
// balanced's, of broad's -- the top-anchored monotonicity.
TEST(VectorCsls, zCutGradedTailMonotone) {
  std::string base = buildAngleFixtureIndex("zgrade", {0.f, 45.f, 90.f},
                                            /*cslsNeighbors=*/3,
                                            /*csls=*/false);
  VectorIndex idx;
  idx.open(base, "zgrade");
  // A head stepping down in ~1-sigma increments from the best (0.90) above a
  // background near 0.50 (spread -> sigma ~ 0.03): Delta 1 keeps the top ~2,
  // Delta 2 ~4, Delta 3 all 5 head members.
  std::vector<float> cos{0.90f, 0.88f, 0.86f, 0.84f, 0.82f, 0.46f,
                         0.48f, 0.50f, 0.50f, 0.52f, 0.54f};
  std::vector<uint64_t> ids;
  for (uint64_t i = 0; i < cos.size(); ++i) ids.push_back(20 + i);
  auto r = makeReranked(cos, ids);
  auto precise =
      keptIds(idx.applyCslsCut(r, cosineZCut(CoverageMode::Precise)));
  auto balanced =
      keptIds(idx.applyCslsCut(r, cosineZCut(CoverageMode::Balanced)));
  auto broad = keptIds(idx.applyCslsCut(r, cosineZCut(CoverageMode::Broad)));
  EXPECT_LE(precise.size(), balanced.size());
  EXPECT_LE(balanced.size(), broad.size());
  EXPECT_LT(precise.size(), broad.size());  // the bands actually differentiate
  EXPECT_TRUE(std::includes(balanced.begin(), balanced.end(), precise.begin(),
                            precise.end()));
  EXPECT_TRUE(std::includes(broad.begin(), broad.end(), balanced.begin(),
                            balanced.end()));
  cleanupTmp(base, "zgrade");
}

// THE REGRESSION this round fixes: a SMOOTH cross-modal relevance gradient
// (one clear best + a long, dense, monotone tail of mildly-related, with NO
// flat background -- the text->image "mona lisa" shape). The old floor-anchored
// `floor + Z*sigma` cut swept up a huge fraction of the tail; the top-anchored
// `s_max - Delta*sigma` cut keeps only a small BOUNDED band below the best.
TEST(VectorCsls, zCutSmoothGradientBoundedKeep) {
  std::string base = buildAngleFixtureIndex("zsmooth", {0.f, 45.f, 90.f},
                                            /*cslsNeighbors=*/3,
                                            /*csls=*/false);
  VectorIndex idx;
  idx.open(base, "zsmooth");
  // One clear best (0.30), then a dense smooth gradient 0.22 -> 0.06 with no
  // gap/plateau (250 points).
  constexpr size_t N = 250;
  std::vector<float> cos;
  std::vector<uint64_t> ids;
  cos.push_back(0.30f);
  ids.push_back(500);
  for (size_t i = 0; i < N; ++i) {
    cos.push_back(0.22f -
                  0.16f * static_cast<float>(i) / static_cast<float>(N));
    ids.push_back(501 + i);
  }
  auto r = makeReranked(cos, ids);
  auto broad = idx.applyCslsCut(r, cosineZCut(CoverageMode::Broad));
  auto precise = idx.applyCslsCut(r, cosineZCut(CoverageMode::Precise));
  // The best is on top, and even the widest mode keeps only a small band --
  // NOT the whole smooth tail (the over-return the old floor cut produced).
  ASSERT_GE(precise.size(), 1u);
  EXPECT_EQ(precise[0].entity_, mkId(500));
  EXPECT_EQ(broad[0].entity_, mkId(500));
  EXPECT_LT(broad.size(), cos.size() / 4);  // bounded, not the whole gradient
  EXPECT_LE(precise.size(), broad.size());
  cleanupTmp(base, "zsmooth");
}

// A flat / no-standout field: `exact` returns NOTHING (the no-match answer);
// every other mode returns exactly the single best.
TEST(VectorCsls, zCutFlatFieldNoMatch) {
  std::string base = buildAngleFixtureIndex("zflat", {0.f, 45.f, 90.f},
                                            /*cslsNeighbors=*/3,
                                            /*csls=*/false);
  VectorIndex idx;
  idx.open(base, "zflat");
  // A flat field: the reranked window has no distinguishable structure (the
  // no-match shape -- the coarse layer cannot separate a background), so the
  // estimated spread collapses and nothing clears the bar.
  auto r = makeReranked({0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f},
                        {30, 31, 32, 33, 34, 35});
  EXPECT_EQ(idx.applyCslsCut(r, cosineZCut(CoverageMode::Exact)).size(), 0u);
  for (CoverageMode m :
       {CoverageMode::Precise, CoverageMode::Balanced, CoverageMode::Broad}) {
    auto hits = idx.applyCslsCut(r, cosineZCut(m));
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].entity_, mkId(30));  // the single best (ties by id)
  }
  cleanupTmp(base, "zflat");
}

// Adaptive rerank: a WIDE top cluster (matches beyond the initial batch)
// forces the top-anchored widen loop to extend past it and recover the cluster
// (the "statues" shape); a SMOOTH gradient does NOT chase to the cap.
TEST(VectorCsls, zCutAdaptiveRerankPlateau) {
  // A two-layer (i8 + bf16) index with a near cluster + random background.
  constexpr size_t N = 800;
  constexpr size_t NEAR = 20;
  constexpr size_t D = 512;
  std::mt19937 rng{20260710};
  std::normal_distribution<float> g{0.f, 1.f};
  auto randomUnit = [&] {
    std::vector<float> v(D);
    float norm = 0;
    for (auto& x : v) {
      x = g(rng);
      norm += x * x;
    }
    norm = std::sqrt(norm);
    for (auto& x : v) x /= norm;
    return v;
  };
  const std::vector<float> center = randomUnit();
  auto nearCenter = [&](float deg) {
    std::vector<float> w = randomUnit();
    float dot = 0;
    for (size_t j = 0; j < D; ++j) dot += w[j] * center[j];
    float norm = 0;
    for (size_t j = 0; j < D; ++j) {
      w[j] -= dot * center[j];
      norm += w[j] * w[j];
    }
    norm = std::sqrt(norm);
    const float a = deg * std::numbers::pi_v<float> / 180.f;
    std::vector<float> v(D);
    for (size_t j = 0; j < D; ++j)
      v[j] = std::cos(a) * center[j] + std::sin(a) * w[j] / norm;
    return v;
  };
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = "zadapt";
  cfg.dimensions_ = D;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.csls_ = false;
  // i8 coarse (a clean quantized-cosine ranking, unlike the noisier 1-bit
  // binary at this small dimension) so the near cluster is coarse-best.
  cfg.scalar_ = VectorScalar::I8;
  cfg.rerankScalar_ = VectorScalar::Bf16;
  VectorIndexBuilder builder{basename, cfg};
  std::uniform_real_distribution<float> clusterAngle{2.f, 7.f};
  for (size_t i = 0; i < N; ++i) {
    std::vector<float> v =
        i < NEAR ? nearCenter(clusterAngle(rng)) : randomUnit();
    builder.add(mkId(2000 + i), "<http://ex/z" + std::to_string(i) + ">", v);
  }
  builder.build();
  VectorIndex idx;
  idx.open(basename, "zadapt");

  const float ff = DEFAULT_ZCUT_FLOOR_FRACTION;
  const float wd = DEFAULT_ZCUT_DELTA_BROAD + DEFAULT_ZCUT_WIDEN_MARGIN;
  // A batch smaller than the cluster (the cluster of 20 lies beyond it) forces
  // the widen loop to extend past the first batch.
  idx.setCslsRerankFloor(8);
  auto shallow = idx.computeCslsReranked(center, 10, ff, wd, /*rerankCap=*/0);
  EXPECT_GT(shallow.rerankDepth(), 8u);  // widened past the initial batch
  EXPECT_TRUE(shallow.plateauFound_);    // stopped, did not chase the cap
  EXPECT_LT(shallow.rerankDepth(), N);
  // Full-depth reference recovers the whole near cluster (the real matches).
  idx.setCslsRerankFloor(N);
  auto full = idx.computeCslsReranked(center, 10, ff, wd, /*rerankCap=*/0);
  ASSERT_EQ(full.rerankDepth(), N);
  std::set<uint64_t> nearBits;
  for (uint64_t i = 0; i < NEAR; ++i) nearBits.insert(mkId(2000 + i).getBits());
  auto nearRecovered = [&](const std::set<uint64_t>& s) {
    size_t hit = 0;
    for (uint64_t b : nearBits) hit += s.count(b);
    return hit;
  };
  for (CoverageMode m : {CoverageMode::Balanced, CoverageMode::Broad}) {
    auto a = keptIds(idx.applyCslsCut(shallow, cosineZCut(m)));
    auto b = keptIds(idx.applyCslsCut(full, cosineZCut(m)));
    EXPECT_GE(nearRecovered(b), NEAR - 3) << "mode " << static_cast<int>(m);
    // The adaptive (shallow) cut keeps only cluster members and never anything
    // the full-depth cut would not keep. (With the production-sized initial
    // batch the first window already holds real background, so shallow == full
    // there; the tiny batch here only narrows the window's sigma estimate.)
    EXPECT_TRUE(std::includes(b.begin(), b.end(), a.begin(), a.end()))
        << "mode " << static_cast<int>(m);
    EXPECT_EQ(nearRecovered(a), a.size());  // nothing but cluster members
  }

  // A SMOOTH gradient (angles 1..90 degrees around a fresh center, no cluster)
  // must NOT chase the rerank cap: the window separates from the best after a
  // few batches, so it stops well short of the cap.
  {
    std::string gname = uniqueTmpBasename();
    VectorIndexConfig gcfg;
    gcfg.name_ = "zgrad";
    gcfg.dimensions_ = D;
    gcfg.metric_ = VectorMetric::Cosine;
    gcfg.buildHnsw_ = false;
    gcfg.scalar_ = VectorScalar::I8;
    gcfg.rerankScalar_ = VectorScalar::Bf16;
    const std::vector<float> gcenter = randomUnit();
    VectorIndexBuilder gb{gname, gcfg};
    constexpr size_t GN = 600;
    for (size_t i = 0; i < GN; ++i) {
      // A dense smooth spread of angles 1..80 degrees -- one best, no plateau.
      gb.add(mkId(9000 + i), "<http://ex/g" + std::to_string(i) + ">",
             nearCenter(1.f +
                        79.f * static_cast<float>(i) / static_cast<float>(GN)));
    }
    gb.build();
    VectorIndex gidx;
    gidx.open(gname, "zgrad");
    gidx.setCslsRerankFloor(8);
    const size_t cap = 8 * 8;  // resolveCslsCut's cslsRerankFloor * 8
    auto gr = gidx.computeCslsReranked(gcenter, 10, ff, wd, cap);
    EXPECT_TRUE(gr.plateauFound_);     // stopped, did NOT hit the cap
    EXPECT_LT(gr.rerankDepth(), cap);  // well short of the cap
    cleanupTmp(gname, "zgrad");
  }
  cleanupTmp(basename, "zadapt");
}

// _____________________________________________________________________________
// The SCORE CACHE: switching the coverage mode / signal on a repeat query
// re-applies the cut over a CACHED reranked stage (no rescan). Distinct
// coverage modes over the SAME query point share ONE reranked-cache entry.
TEST(VectorCsls, autoCutModeSwitchHitsScoreCache) {
  auto* qec = qecWithCslsIndexes();
  auto getId = makeGetId(qec->getIndex());
  qlever::vector::clearCslsRerankedCacheForTesting();
  auto run = [&](std::string_view autoCut, std::string_view signal) {
    std::string sig = signal.empty()
                          ? std::string{}
                          : absl::StrCat("vec:cutSignal \"", signal, "\" ; ");
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embc\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; " +
                 sig + "vec:autoCut \"" + std::string{autoCut} + "\" . } }");
    auto result = qet.getResult();
    std::vector<Id> ids;
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    for (size_t r = 0; r < result->idTable().numRows(); ++r)
      ids.push_back(result->idTable()(r, nnCol));
    return ids;
  };
  // First mode: a cache MISS populates one reranked entry.
  auto precise = run("precise", "cosine");
  EXPECT_EQ(qlever::vector::cslsRerankedCacheSizeForTesting(), 1u);
  // A different coverage mode over the same point: HIT (still one entry), and
  // the broader net keeps at least as many as the precise one.
  auto broad = run("broad", "cosine");
  EXPECT_EQ(qlever::vector::cslsRerankedCacheSizeForTesting(), 1u);
  // A different SIGNAL over the same point: also mode-independent -> HIT.
  auto csls = run("balanced", "csls");
  EXPECT_EQ(qlever::vector::cslsRerankedCacheSizeForTesting(), 1u);
  // The re-applied cuts are real results (the standout d on top, broad >=
  // precise).
  ASSERT_GE(precise.size(), 1u);
  ASSERT_GE(broad.size(), 1u);
  ASSERT_GE(csls.size(), 1u);
  EXPECT_EQ(precise[0], getId("<d>"));
  EXPECT_EQ(broad[0], getId("<d>"));
  EXPECT_EQ(csls[0], getId("<d>"));
  EXPECT_GE(broad.size(), precise.size());
}
