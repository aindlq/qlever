// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests of the TWO-LAYER quantize+rerank vector store: the `rerank` build key
// (a coarse i8 scan matrix `.data` plus a fine bf16 rerank matrix
// `.rerank.data` from the same input), per-layer RAM residency
// (`preload`/`preloadRerank`), the fine-layer exact baseline (`vec:distance`
// reads the rerank matrix), and the SERVICE's coarse-scan-then-rerank top-k
// (`vec:rerankK`, `vec:bindScore` = fine, `vec:bindCoarseScore` = coarse).
// Also covers the `vec:candidates` rename of `vec:left` (alias kept).

#include <absl/base/casts.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "./util/GTestHelpers.h"
#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "global/Id.h"
#include "global/IndexTypes.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "parser/SparqlParser.h"
#include "services/vectorSearch/VectorBf16Kernels.h"
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

// The test knowledge graph. `<r2>` is additionally a `<SeedItem>`, the outer
// binding used by the `vec:candidates`/`vec:left` alias test.
constexpr std::string_view KG_RERANK =
    "<r0> <is-a> <RItem> . <r1> <is-a> <RItem> . <r2> <is-a> <RItem> . "
    "<r3> <is-a> <RItem> . <r4> <is-a> <RItem> . <r5> <is-a> <RItem> . "
    "<r2> <is-a> <SeedItem> .";

// Dimension-4 vectors, cosine metric. Against the query [0,1,0,0] the FINE
// (bf16) ranking is unambiguous: r3 (0) < r2 (0.2) < r1 (0.4) < r5 (~0.997)
// < r0 = r4 (1). The components of r0..r4 are i8-friendly, so the coarse i8
// top-`rerankK` preserves the fine top set. `<r5>` = [1, 0.003, 0, 0] is the
// designed QUANTIZATION-ERROR probe: usearch's i8 cast (normalize, scale to
// +-127, truncate) drops the 0.003 component entirely (0.003 * 127 < 1), so
// its coarse distance to [0,1,0,0] is exactly 1 while the fine bf16 distance
// is ~0.997 -- a >= 1e-3 gap every layer-confusion bug would erase.
const std::vector<std::pair<std::string, std::vector<float>>>&
rerankTestVectors() {
  static const std::vector<std::pair<std::string, std::vector<float>>> vecs{
      {"<r0>", {1.f, 0.f, 0.f, 0.f}},   {"<r1>", {0.8f, 0.6f, 0.f, 0.f}},
      {"<r2>", {0.6f, 0.8f, 0.f, 0.f}}, {"<r3>", {0.f, 1.f, 0.f, 0.f}},
      {"<r4>", {0.f, 0.f, 1.f, 0.f}},   {"<r5>", {1.f, 0.003f, 0.f, 0.f}},
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

// The shared test context: FOUR indices over the same vectors, built through
// the REGISTERED build/load hooks (the `--service-index` path incl. the
// `rerank` key of `parseSpec`):
//   * "embrr"  -- the two-layer index under test (scan i8 + rerank bf16);
//   * "embbf"  -- a single-layer bf16 reference (the fine layer's twin);
//   * "embi8"  -- a single-layer i8 reference (the coarse layer's twin);
//   * "embbin" -- a two-layer BINARY index (scan 1-bit sign-packed Hamming +
//                 rerank bf16). Against the query [0,1,0,0] the sign patterns
//                 ({0} for r0, {0,1} for r1/r2/r5, {1} for r3, {2} for r4 vs
//                 the query's {1}) put the whole fine top set within the wide
//                 binary default rerankK, so the reranked top-k must equal
//                 the exact fine top-k.
QueryExecutionContext* qecWithRerankIndexes() {
  QueryExecutionContext* qec = getQec(std::string{KG_RERANK});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{VECTOR_EXTENSION_NAME}) != nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecreranktest-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, rerankTestVectors());
  auto entry = [&](const std::string& name, const char* scalar,
                   const char* rerank) {
    nlohmann::json e{{"name", name},       {"npy", npy},       {"iris", iris},
                     {"metric", "cosine"}, {"scalar", scalar}, {"hnsw", false}};
    if (rerank != nullptr) {
      e["rerank"] = rerank;
    }
    return e;
  };
  nlohmann::json spec{
      {"vectorSearch",
       nlohmann::json::array({entry("embrr", "i8", "bf16"),
                              entry("embbf", "bf16", nullptr),
                              entry("embi8", "i8", nullptr),
                              entry("embbin", "binary", "bf16")})}};
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  // The two-layer builds must produce the rerank matrix ONLY for "embrr" and
  // "embbin".
  EXPECT_TRUE(std::filesystem::exists(vectorRerankDataFile(basename, "embrr")));
  EXPECT_TRUE(
      std::filesystem::exists(vectorRerankDataFile(basename, "embbin")));
  EXPECT_FALSE(
      std::filesystem::exists(vectorRerankDataFile(basename, "embbf")));
  EXPECT_FALSE(
      std::filesystem::exists(vectorRerankDataFile(basename, "embi8")));
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  // The load hook memory-maps everything; the directory entries can go.
  for (std::string_view name : {"embrr", "embbf", "embi8", "embbin"}) {
    for (std::string_view suffix : {".meta", ".keys", ".rowmap", ".data",
                                    ".rerank.data", ".iris", ".hnsw"}) {
      std::error_code ec;
      std::filesystem::remove(
          basename + ".vec." + std::string{name} + std::string{suffix}, ec);
    }
  }
  for (std::string_view suffix : {".input.npy", ".input.iris"}) {
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
          (std::string{"qlever-vecreranktest-"} + info->test_suite_name() +
           "-" + info->name()))
      .string();
}

Id mkId(uint64_t v) { return Id::makeFromVocabIndex(VocabIndex::make(v)); }

// The dimension of the direct-builder fixture: large enough that the
// page-rounded on-disk sizes of the 1-byte i8 scan matrix and the 2-byte bf16
// rerank matrix land in DISJOINT ranges (both matrices hold 4 rows).
constexpr size_t BIG_DIM = 4096;
constexpr size_t BIG_ROWS = 4;

// Row `i` is the (bf16-exact) two-hot vector 0.5*(e_{2i} + e_{2i+1}).
std::vector<float> bigRow(size_t i) {
  std::vector<float> v(BIG_DIM, 0.f);
  v[2 * i] = 0.5f;
  v[2 * i + 1] = 0.5f;
  return v;
}

// Build a two-layer (or, with `rerank = nullopt`, single-layer) tmp index.
std::string buildTmpTwoLayer(std::optional<VectorScalar> rerank,
                             std::string name = "rr",
                             VectorScalar scan = VectorScalar::I8) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = BIG_DIM;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = scan;
  cfg.rerankScalar_ = rerank;
  cfg.buildHnsw_ = false;
  VectorIndexBuilder builder{basename, cfg};
  for (size_t i = 0; i < BIG_ROWS; ++i) {
    builder.add(mkId(100 + i), "<http://ex/" + std::to_string(i) + ">",
                bigRow(i));
  }
  builder.build();
  return basename;
}

void cleanupTmp(const std::string& basename, const std::string& name = "rr") {
  for (std::string_view suffix : {".meta", ".keys", ".rowmap", ".data",
                                  ".rerank.data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec." + name + std::string{suffix},
                            ec);
  }
}

nlohmann::json readMetaJson(const std::string& basename,
                            const std::string& name = "rr") {
  std::ifstream in{vectorMetaFile(basename, name)};
  EXPECT_TRUE(in.is_open());
  nlohmann::json j;
  in >> j;
  return j;
}
}  // namespace

// _____________________________________________________________________________
// The two-layer build writes BOTH matrices -- the coarse i8 `.data` and the
// fine bf16 `.rerank.data`, same rows, same order -- and records the rerank
// precision in the `.meta`. The fine layer is what `getVector` decodes: the
// bf16-exact input components come back bit-exactly, which the normalized,
// truncated i8 scan bytes could never reproduce.
TEST(VectorRerank, buildWritesBothMatricesAndMeta) {
  std::string basename = buildTmpTwoLayer(VectorScalar::Bf16);

  // Both matrices exist, with their layer's byte size (payload up to one page
  // of `MmapVector` rounding + its trailer). i8 = 1 B/dim, bf16 = 2 B/dim, so
  // the two size ranges are disjoint at this dimension.
  const auto pageSlack = static_cast<std::uintmax_t>(getpagesize()) + 32;
  const std::uintmax_t scanBytes =
      std::filesystem::file_size(vectorDataFile(basename, "rr"));
  const std::uintmax_t rerankBytes =
      std::filesystem::file_size(vectorRerankDataFile(basename, "rr"));
  EXPECT_GE(scanBytes, BIG_ROWS * BIG_DIM);
  EXPECT_LE(scanBytes, BIG_ROWS * BIG_DIM + pageSlack);
  EXPECT_GE(rerankBytes, BIG_ROWS * BIG_DIM * 2);
  EXPECT_LE(rerankBytes, BIG_ROWS * BIG_DIM * 2 + pageSlack);
  EXPECT_GT(rerankBytes, scanBytes);

  // The `.meta` records both precisions.
  nlohmann::json meta = readMetaJson(basename);
  EXPECT_EQ(meta.at("scalar"), "i8");
  ASSERT_TRUE(meta.contains("rerankScalar"));
  EXPECT_EQ(meta.at("rerankScalar"), "bf16");

  VectorIndex idx;
  idx.open(basename, "rr");
  EXPECT_TRUE(idx.hasRerankLayer());
  ASSERT_TRUE(idx.metadata().config_.rerankScalar_.has_value());
  EXPECT_EQ(idx.metadata().config_.rerankScalar_.value(), VectorScalar::Bf16);

  // `getVector` decodes the FINE layer: the bf16-exact 0.5 components come
  // back exactly. (The i8 layer stores the normalized row [~89, ~89, 0, ...],
  // which decodes to ~0.70 -- unmistakably different.)
  auto v = idx.getVector(mkId(101));
  ASSERT_TRUE(v.has_value());
  EXPECT_FLOAT_EQ((*v)[2], 0.5f);
  EXPECT_FLOAT_EQ((*v)[3], 0.5f);
  EXPECT_FLOAT_EQ((*v)[0], 0.f);

  // The exact search runs on the fine layer and finds each row's self.
  auto hits = idx.searchExact(bigRow(1), 1);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].entity_, mkId(101));
  EXPECT_NEAR(hits[0].distance_, 0.f, 1e-4);
  cleanupTmp(basename);
}

// _____________________________________________________________________________
// The BINARY two-layer build: the coarse scan matrix holds 1-bit sign-packed
// rows of `(dim + 7) / 8` bytes (the 32x storage rung: 512 B per 4096-dim
// vector), the fine bf16 rerank matrix is written exactly as for i8, and the
// `.meta` records scalar=binary + rerankScalar=bf16. Exact reads (`getVector`,
// `searchExact`) serve the fine layer; the coarse layer serves integer
// Hamming distances.
TEST(VectorRerank, binaryBuildWritesBothMatricesAndMeta) {
  std::string basename =
      buildTmpTwoLayer(VectorScalar::Bf16, "rr", VectorScalar::Binary);

  // Scan payload: BIG_ROWS * BIG_DIM / 8 bytes (1 bit per component), far
  // below the bf16 rerank matrix's 2 bytes per component.
  const auto pageSlack = static_cast<std::uintmax_t>(getpagesize()) + 32;
  const std::uintmax_t scanBytes =
      std::filesystem::file_size(vectorDataFile(basename, "rr"));
  const std::uintmax_t rerankBytes =
      std::filesystem::file_size(vectorRerankDataFile(basename, "rr"));
  EXPECT_GE(scanBytes, BIG_ROWS * BIG_DIM / 8);
  EXPECT_LE(scanBytes, BIG_ROWS * BIG_DIM / 8 + pageSlack);
  EXPECT_GE(rerankBytes, BIG_ROWS * BIG_DIM * 2);
  EXPECT_LE(rerankBytes, BIG_ROWS * BIG_DIM * 2 + pageSlack);

  nlohmann::json meta = readMetaJson(basename);
  EXPECT_EQ(meta.at("scalar"), "binary");
  ASSERT_TRUE(meta.contains("rerankScalar"));
  EXPECT_EQ(meta.at("rerankScalar"), "bf16");
  EXPECT_EQ(meta.at("rowStrideBytes"), BIG_DIM / 8);

  VectorIndex idx;
  idx.open(basename, "rr");
  EXPECT_TRUE(idx.hasRerankLayer());
  EXPECT_EQ(idx.metadata().config_.scalar_, VectorScalar::Binary);

  // `getVector` decodes the FINE bf16 layer: the 0.5 components come back
  // exactly (the packed sign bits could only yield 0/1).
  auto v = idx.getVector(mkId(101));
  ASSERT_TRUE(v.has_value());
  EXPECT_FLOAT_EQ((*v)[2], 0.5f);
  EXPECT_FLOAT_EQ((*v)[3], 0.5f);
  EXPECT_FLOAT_EQ((*v)[0], 0.f);

  // The exact search reads the fine layer: self at cosine distance ~0.
  auto hits = idx.searchExact(bigRow(1), 1);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].entity_, mkId(101));
  EXPECT_NEAR(hits[0].distance_, 0.f, 1e-4);

  // The coarse layer serves HAMMING distances: the two-hot sign patterns of
  // distinct rows are disjoint, so self = 0 and every other row = 4.
  auto coarse = idx.searchExactCoarse(bigRow(1), BIG_ROWS);
  ASSERT_EQ(coarse.size(), BIG_ROWS);
  EXPECT_EQ(coarse[0].entity_, mkId(101));
  EXPECT_EQ(coarse[0].distance_, 0.f);
  for (size_t i = 1; i < BIG_ROWS; ++i) {
    EXPECT_EQ(coarse[i].distance_, 4.f) << i;
  }
  cleanupTmp(basename);
}

// _____________________________________________________________________________
// A single-layer build produces exactly the OLD `.meta` format -- no
// `rerankScalar` key, no `.rerank.data` -- and it (i.e. any pre-two-layer
// index) still loads as a single-layer index.
TEST(VectorRerank, singleLayerMetaHasNoRerankFieldAndLoads) {
  std::string basename = buildTmpTwoLayer(std::nullopt);
  EXPECT_FALSE(std::filesystem::exists(vectorRerankDataFile(basename, "rr")));
  nlohmann::json meta = readMetaJson(basename);
  EXPECT_FALSE(meta.contains("rerankScalar"));

  VectorIndex idx;
  idx.open(basename, "rr");
  EXPECT_FALSE(idx.hasRerankLayer());
  EXPECT_FALSE(idx.metadata().config_.rerankScalar_.has_value());
  EXPECT_EQ(idx.rerankResidency(), VectorIndex::Residency::None);
  auto hits = idx.searchExact(bigRow(2), 1);
  ASSERT_EQ(hits.size(), 1u);
  EXPECT_EQ(hits[0].entity_, mkId(102));
  cleanupTmp(basename);
}

// _____________________________________________________________________________
// The rerank layer must be a HIGH-precision scalar: `i8` is rejected both by
// the build-spec parser (`parseSpec`, reached through the registered build
// hook) and by the builder itself.
TEST(VectorRerank, i8RerankLayerIsRejected) {
  // Builder-level (the spec parser never lets an i8 rerank through, but the
  // builder guards direct users too).
  VectorIndexConfig cfg;
  cfg.name_ = "bad";
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = VectorScalar::I8;
  cfg.rerankScalar_ = VectorScalar::I8;
  AD_EXPECT_THROW_WITH_MESSAGE((VectorIndexBuilder{uniqueTmpBasename(), cfg}),
                               HasSubstr("rerank layer"));

  // Spec-level, through the registered build hook (`parseSpec` throws before
  // any input file is touched, so the bogus paths never matter).
  auto* qec = qecWithRerankIndexes();
  ASSERT_FALSE(IndexExtensionRegistry::get().buildHooks().empty());
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          qec->getIndex(), "/nonexistent/base",
          nlohmann::json::parse(
              R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
              R"("iris":"/nonexistent.iris","scalar":"i8","rerank":"i8"}]})")),
      HasSubstr("rerank"));
}

// _____________________________________________________________________________
// `binary` is a SCAN-layer-only precision: `rerank: "binary"` is rejected both
// by the build-spec parser and by the builder itself (the rerank layer is the
// high-precision layer that exact distances read).
TEST(VectorRerank, binaryRerankLayerIsRejected) {
  VectorIndexConfig cfg;
  cfg.name_ = "bad";
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = VectorScalar::Binary;
  cfg.rerankScalar_ = VectorScalar::Binary;
  AD_EXPECT_THROW_WITH_MESSAGE((VectorIndexBuilder{uniqueTmpBasename(), cfg}),
                               HasSubstr("rerank layer"));

  auto* qec = qecWithRerankIndexes();
  ASSERT_FALSE(IndexExtensionRegistry::get().buildHooks().empty());
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          qec->getIndex(), "/nonexistent/base",
          nlohmann::json::parse(
              R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
              R"("iris":"/nonexistent.iris","scalar":"binary",)"
              R"("rerank":"binary"}]})")),
      HasSubstr("rerank"));
}

// _____________________________________________________________________________
// `binary` keeps only sign bits (Hamming, an angular proxy), so like `i8` it
// is rejected for metrics other than cosine.
TEST(VectorRerank, binaryRejectsNonCosineMetric) {
  auto* qec = qecWithRerankIndexes();
  ASSERT_FALSE(IndexExtensionRegistry::get().buildHooks().empty());
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          qec->getIndex(), "/nonexistent/base",
          nlohmann::json::parse(
              R"({"vectorSearch":[{"name":"q","npy":"/nonexistent.npy",)"
              R"("iris":"/nonexistent.iris","scalar":"binary",)"
              R"("metric":"l2sq"}]})")),
      HasSubstr("cosine"));
}

// _____________________________________________________________________________
// Per-layer residency: `preload` governs the scan matrix, `preloadRerank` the
// rerank matrix, independently, at open time. The index answers correctly
// under every combination (residency never affects results).
TEST(VectorRerank, perLayerResidency) {
  std::string basename = buildTmpTwoLayer(VectorScalar::Bf16);

  // The operator's headline combination: pin the small i8 scan matrix, leave
  // the bf16 rerank matrix demand-paged.
  {
    VectorIndex idx;
    idx.open(basename, "rr", VectorIndex::Residency::Lock,
             VectorIndex::Residency::None);
    EXPECT_EQ(idx.residency(), VectorIndex::Residency::Lock);
    EXPECT_EQ(idx.rerankResidency(), VectorIndex::Residency::None);
    auto hits = idx.searchExact(bigRow(0), 2);
    ASSERT_EQ(hits.size(), 2u);
    EXPECT_EQ(hits[0].entity_, mkId(100));
    EXPECT_NEAR(hits[0].distance_, 0.f, 1e-4);
  }
  // The reverse: rerank matrix preloaded, scan matrix plain mmap.
  {
    VectorIndex idx;
    idx.open(basename, "rr", VectorIndex::Residency::None,
             VectorIndex::Residency::Advise);
    EXPECT_EQ(idx.residency(), VectorIndex::Residency::None);
    EXPECT_EQ(idx.rerankResidency(), VectorIndex::Residency::Advise);
    auto hits = idx.searchExactCoarse(bigRow(0), 1);
    ASSERT_EQ(hits.size(), 1u);
    EXPECT_EQ(hits[0].entity_, mkId(100));
  }
  cleanupTmp(basename);
}

// _____________________________________________________________________________
// The `preloadRerank` field of `QLEVER_VECTOR_SEARCH_ENDPOINTS` parses next to
// `preload` (same value set); an invalid value poisons the whole entry.
TEST(VectorRerank, preloadRerankEnvOverrideParses) {
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"a": {"preload": "lock", "preloadRerank": "none"},)"
      R"( "b": {"preloadRerank": "advise"},)"
      R"( "bad": {"preload": "lock", "preloadRerank": "locked"}})");
  ASSERT_EQ(overrides.size(), 2u);
  EXPECT_EQ(overrides.at("a").preload_, "lock");
  EXPECT_EQ(overrides.at("a").preloadRerank_, "none");
  EXPECT_FALSE(overrides.at("b").preload_.has_value());
  EXPECT_EQ(overrides.at("b").preloadRerank_, "advise");
  EXPECT_FALSE(overrides.contains("bad"));
}

// _____________________________________________________________________________
// The SERVICE's coarse-scan-then-rerank top-k: identical -- entities AND fine
// distances -- to the exact fine-layer `searchExact` top-k, with
// `vec:bindScore` the fine distance and `vec:bindCoarseScore` the coarse scan
// distance (which visibly differs on the designed `<r5>`).
TEST(VectorRerank, serviceRerankMatchesExactFineSearch) {
  auto* qec = qecWithRerankIndexes();
  auto vidx = getVectorIndex(qec->getIndex(), "embrr");
  ASSERT_TRUE(vidx != nullptr);
  ASSERT_TRUE(vidx->hasRerankLayer());
  const std::vector<float> query{0.f, 1.f, 0.f, 0.f};

  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embrr\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
               "vec:bindScore ?d ; vec:bindCoarseScore ?dc ; vec:k 3 . } }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  const IdTable& table = result->idTable();

  // Reference: the exact fine-layer top-3 (the same primitive `vec:distance`
  // ranks with) and the coarse distances of ALL entities.
  auto fineTop = vidx->searchExact(query, 3);
  ASSERT_EQ(fineTop.size(), 3u);
  auto coarseAll = vidx->searchExactCoarse(query, vidx->numLiveVectors());
  ad_utility::HashMap<Id, float> coarseByEntity;
  for (const auto& hit : coarseAll) {
    coarseByEntity[hit.entity_] = hit.distance_;
  }

  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(table.numRows(), 3u);
  // Unambiguous fine ranking: r3 (0) < r2 (0.2) < r1 (0.4).
  EXPECT_EQ(table(0, nnCol), getId("<r3>"));
  EXPECT_EQ(table(1, nnCol), getId("<r2>"));
  EXPECT_EQ(table(2, nnCol), getId("<r1>"));
  for (size_t r = 0; r < 3; ++r) {
    EXPECT_EQ(table(r, nnCol), fineTop[r].entity_) << "row " << r;
    // The bound fine score is BIT-IDENTICAL to the exact fine search's.
    EXPECT_EQ(table(r, dCol),
              Id::makeFromDouble(static_cast<double>(fineTop[r].distance_)))
        << "row " << r;
    // The bound coarse score is the scan-layer distance of that entity.
    ASSERT_TRUE(coarseByEntity.contains(table(r, nnCol)));
    EXPECT_EQ(table(r, dcCol), Id::makeFromDouble(static_cast<double>(
                                   coarseByEntity[table(r, nnCol)])))
        << "row " << r;
  }
}

// _____________________________________________________________________________
// The BINARY SERVICE top-k: the coarse pass ranks by HAMMING distance over
// the sign bits, the fine pass rescores exactly on bf16. On this fixture the
// sign patterns keep the whole fine top set within the (wide) binary default
// rerankK, so the binary-scan -> bf16-rerank top-k EQUALS the exact bf16
// `searchExact` top-k -- entities and bit-identical fine scores, with
// `vec:bindScore` == the exact `vec:distance`. `vec:bindCoarseScore` binds
// the raw Hamming distance: an INTEGER in [0, dim] on a different scale than
// the fine cosine score (deliberately not reconciled -- `ABS(?d - ?dc)` is
// meaningless here, unlike on the i8 index).
TEST(VectorRerank, serviceBinaryRerankMatchesExactFineSearch) {
  auto* qec = qecWithRerankIndexes();
  auto vidx = getVectorIndex(qec->getIndex(), "embbin");
  ASSERT_TRUE(vidx != nullptr);
  ASSERT_TRUE(vidx->hasRerankLayer());
  ASSERT_EQ(vidx->metadata().config_.scalar_, VectorScalar::Binary);
  const std::vector<float> query{0.f, 1.f, 0.f, 0.f};

  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embbin\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
               "vec:bindScore ?d ; vec:bindCoarseScore ?dc ; vec:k 3 . } "
               "BIND(vec:distance(vidx:embbin, ?nn, \"0,1,0,0\") AS ?dref) }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  size_t drefCol = qet.getVariableColumn(Variable{"?dref"});
  const IdTable& table = result->idTable();

  // Reference: the exact fine-layer (bf16) top-3 and the coarse Hamming
  // distances of ALL entities.
  auto fineTop = vidx->searchExact(query, 3);
  ASSERT_EQ(fineTop.size(), 3u);
  auto coarseAll = vidx->searchExactCoarse(query, vidx->numLiveVectors());
  ad_utility::HashMap<Id, float> coarseByEntity;
  for (const auto& hit : coarseAll) {
    coarseByEntity[hit.entity_] = hit.distance_;
  }

  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(table.numRows(), 3u);
  // Unambiguous fine ranking: r3 (0) < r2 (0.2) < r1 (0.4) -- exactly the
  // exact bf16 top-3.
  EXPECT_EQ(table(0, nnCol), getId("<r3>"));
  EXPECT_EQ(table(1, nnCol), getId("<r2>"));
  EXPECT_EQ(table(2, nnCol), getId("<r1>"));
  for (size_t r = 0; r < 3; ++r) {
    EXPECT_EQ(table(r, nnCol), fineTop[r].entity_) << "row " << r;
    // Fine score: bit-identical to the exact fine search AND to the exact
    // `vec:distance` of the same entity.
    EXPECT_EQ(table(r, dCol),
              Id::makeFromDouble(static_cast<double>(fineTop[r].distance_)))
        << "row " << r;
    EXPECT_EQ(table(r, dCol), table(r, drefCol)) << "row " << r;
    // Coarse score: the raw Hamming distance -- an integer in [0, dim].
    ASSERT_TRUE(coarseByEntity.contains(table(r, nnCol)));
    double hamming = table(r, dcCol).getDouble();
    EXPECT_EQ(table(r, dcCol), Id::makeFromDouble(static_cast<double>(
                                   coarseByEntity[table(r, nnCol)])))
        << "row " << r;
    EXPECT_EQ(hamming, std::floor(hamming)) << "row " << r;
    EXPECT_GE(hamming, 0.0);
    EXPECT_LE(hamming, 4.0);  // dim = 4
  }
  // The designed Hamming values: the query's sign pattern is {1}; r3's is
  // {1} (distance 0), r2's and r1's are {0,1} (distance 1).
  EXPECT_EQ(table(0, dcCol), Id::makeFromDouble(0.0));
  EXPECT_EQ(table(1, dcCol), Id::makeFromDouble(1.0));
  EXPECT_EQ(table(2, dcCol), Id::makeFromDouble(1.0));
}

// _____________________________________________________________________________
// The quantization-error probe: for `<r5>` the fine (bf16) distance and the
// coarse (i8) distance to [0,1,0,0] differ by ~3e-3 (the i8 cast drops the
// 0.003 component), so `ABS(?d - ?dc)` exposes the quantization error, while
// the two scores still agree to <= i8 resolution on the well-quantized rows.
TEST(VectorRerank, coarseScoreExposesQuantizationError) {
  auto* qec = qecWithRerankIndexes();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embrr\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
               "vec:bindScore ?d ; vec:bindCoarseScore ?dc ; vec:k 6 . } }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  ASSERT_EQ(table.numRows(), 6u);
  bool sawR5 = false;
  for (size_t r = 0; r < table.numRows(); ++r) {
    double fine = table(r, dCol).getDouble();
    double coarse = table(r, dcCol).getDouble();
    if (table(r, nnCol) == getId("<r5>")) {
      sawR5 = true;
      // Fine: ~1 - 0.003 (bf16 keeps the small component); coarse: exactly 1
      // (i8 dropped it). The gap is the quantization error.
      EXPECT_NEAR(fine, 0.997, 1e-3);
      EXPECT_NEAR(coarse, 1.0, 1e-6);
      EXPECT_GT(std::abs(fine - coarse), 1e-3);
    } else {
      // The other rows are i8-friendly: both scores agree to i8 resolution.
      EXPECT_NEAR(fine, coarse, 0.02) << "row " << r;
    }
  }
  EXPECT_TRUE(sawR5);
}

// _____________________________________________________________________________
// `vec:bindScore` of the SERVICE == the exact `vec:distance` on the same
// entity (both read the fine layer), asserted end-to-end in one query.
TEST(VectorRerank, serviceScoreEqualsVecDistance) {
  auto* qec = qecWithRerankIndexes();
  QueryExecutionTree qet = planQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embrr\" ; "
          "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; vec:bindScore ?d ; "
          "vec:k 6 . } "
          "BIND(vec:distance(vidx:embrr, ?nn, \"0,1,0,0\") AS ?dref) }");
  auto result = qet.getResult();
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t drefCol = qet.getVariableColumn(Variable{"?dref"});
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 6u);
  for (size_t r = 0; r < table.numRows(); ++r) {
    ASSERT_EQ(table(r, dCol).getDatatype(), Datatype::Double);
    EXPECT_EQ(table(r, dCol), table(r, drefCol)) << "row " << r;
  }
}

// _____________________________________________________________________________
// `vec:distance` reads the FINE layer of a two-layer index: on `<r5>` it is
// bit-identical to the single-layer bf16 reference index and clearly differs
// from the single-layer i8 reference (which is what the coarse layer stores).
TEST(VectorRerank, vecDistanceReadsFineLayer) {
  auto* qec = qecWithRerankIndexes();
  const std::vector<float> query{0.f, 1.f, 0.f, 0.f};
  auto getId = makeGetId(qec->getIndex());
  Id r5 = getId("<r5>");

  // Primitive level: the two-layer DistanceComputer == the bf16 twin's,
  // != the i8 twin's.
  auto twoLayer = getVectorIndex(qec->getIndex(), "embrr");
  auto bf16Ref = getVectorIndex(qec->getIndex(), "embbf");
  auto i8Ref = getVectorIndex(qec->getIndex(), "embi8");
  ASSERT_TRUE(twoLayer && bf16Ref && i8Ref);
  float dTwoLayer = twoLayer->makeDistanceComputer(query)(r5);
  float dBf16 = bf16Ref->makeDistanceComputer(query)(r5);
  float dI8 = i8Ref->makeDistanceComputer(query)(r5);
  EXPECT_EQ(dTwoLayer, dBf16);
  EXPECT_GT(std::abs(dTwoLayer - dI8), 1e-3f);

  // SPARQL level: the same three distances via `vec:distance`.
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { "
               "BIND(vec:distance(vidx:embrr, <r5>, \"0,1,0,0\") AS ?d) "
               "BIND(vec:distance(vidx:embbf, <r5>, \"0,1,0,0\") AS ?dbf) "
               "BIND(vec:distance(vidx:embi8, <r5>, \"0,1,0,0\") AS ?di8) }");
  auto result = qet.getResult();
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 1u);
  Id d = table(0, qet.getVariableColumn(Variable{"?d"}));
  Id dbf = table(0, qet.getVariableColumn(Variable{"?dbf"}));
  Id di8 = table(0, qet.getVariableColumn(Variable{"?di8"}));
  ASSERT_EQ(d.getDatatype(), Datatype::Double);
  EXPECT_EQ(d, dbf);
  EXPECT_GT(std::abs(d.getDouble() - di8.getDouble()), 1e-3);
}

// _____________________________________________________________________________
// An explicit `vec:rerankK` bounds the coarse candidate pass; a value that
// still covers the fine top-k yields the same result as the default.
TEST(VectorRerank, explicitRerankK) {
  auto* qec = qecWithRerankIndexes();
  auto getId = makeGetId(qec->getIndex());
  auto topEntities = [&](std::string_view extra) {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embrr\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; vec:k 3 " +
                 std::string{extra} + " . } }");
    auto result = qet.getResult();
    size_t col = qet.getVariableColumn(Variable{"?nn"});
    std::vector<Id> out;
    for (size_t r = 0; r < result->idTable().numRows(); ++r) {
      out.push_back(result->idTable()(r, col));
    }
    return out;
  };
  std::vector<Id> expected{getId("<r3>"), getId("<r2>"), getId("<r1>")};
  EXPECT_EQ(topEntities(""), expected);
  EXPECT_EQ(topEntities("; vec:rerankK 4"), expected);
  // On a SINGLE-layer index `vec:rerankK` is accepted and ignored.
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embbf\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; vec:k 2 ; "
               "vec:rerankK 50 . } }");
  auto result = qet.getResult();
  size_t col = qet.getVariableColumn(Variable{"?nn"});
  ASSERT_EQ(result->idTable().numRows(), 2u);
  EXPECT_EQ(result->idTable()(0, col), getId("<r3>"));
  EXPECT_EQ(result->idTable()(1, col), getId("<r2>"));
}

// _____________________________________________________________________________
// On a single-layer index the two layers coincide, so `vec:bindCoarseScore`
// binds the same distance as `vec:bindScore`.
TEST(VectorRerank, coarseScoreOnSingleLayerEqualsScore) {
  auto* qec = qecWithRerankIndexes();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embbf\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:result ?nn ; "
               "vec:bindScore ?d ; vec:bindCoarseScore ?dc ; vec:k 4 . } }");
  auto result = qet.getResult();
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 4u);
  for (size_t r = 0; r < table.numRows(); ++r) {
    EXPECT_EQ(table(r, dCol), table(r, dcCol)) << "row " << r;
  }
}

// _____________________________________________________________________________
// `vec:candidates` is the canonical name of the former `vec:left`; the alias
// keeps working and both produce identical results (the outer-bound join
// form: for each `?x`, the k nearest).
TEST(VectorRerank, candidatesAliasMatchesLeft) {
  auto* qec = qecWithRerankIndexes();
  auto run = [&](std::string_view param) {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { ?x <is-a> <SeedItem> . "
                 "SERVICE vec: { _:c vec:index \"embrr\" ; vec:" +
                 std::string{param} +
                 " ?x ; vec:result ?nn ; vec:bindScore ?d ; vec:k 2 . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    std::vector<std::pair<Id, Id>> rows;
    for (size_t r = 0; r < result->idTable().numRows(); ++r) {
      rows.emplace_back(result->idTable()(r, nnCol),
                        result->idTable()(r, dCol));
    }
    return rows;
  };
  auto viaCandidates = run("candidates");
  auto viaLeft = run("left");
  EXPECT_EQ(viaCandidates, viaLeft);
  // The seed <r2> finds itself first, then <r1> (its unambiguous nearest).
  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(viaCandidates.size(), 2u);
  EXPECT_EQ(viaCandidates[0].first, getId("<r2>"));
  EXPECT_EQ(viaCandidates[1].first, getId("<r1>"));
}

// _____________________________________________________________________________
// FORM W spelled with `vec:candidates` (an explicit query point +
// `vec:candidates ?nn ; vec:result ?nn` with `?nn` unbound by the surrounding
// query) runs the same coarse-scan-then-rerank whole-index path as the plain
// `vec:result ?nn` form on a two-layer index -- bit-equal entities, fine
// scores, AND coarse scores.
TEST(VectorRerank, unboundCandidatesRunsWholeIndexRerankPath) {
  auto* qec = qecWithRerankIndexes();
  auto run = [&](std::string_view bindClause) {
    // Force fresh computations for each form.
    qec->getQueryTreeCache().clearAll();
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { SERVICE vec: { _:c vec:index \"embrr\" ; "
                 "vec:queryVector \"0,1,0,0\" ; " +
                 std::string{bindClause} +
                 " ; vec:bindScore ?d ; vec:bindCoarseScore ?dc ; "
                 "vec:k 3 ; vec:rerankK 6 . } }");
    auto result = qet.getResult();
    size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
    std::vector<std::tuple<uint64_t, uint64_t, uint64_t>> rows;
    for (size_t r = 0; r < result->idTable().numRows(); ++r) {
      rows.emplace_back(result->idTable()(r, nnCol).getBits(),
                        result->idTable()(r, dCol).getBits(),
                        result->idTable()(r, dcCol).getBits());
    }
    return rows;
  };
  auto viaCandidates = run("vec:candidates ?nn ; vec:result ?nn");
  auto viaResult = run("vec:result ?nn");
  EXPECT_EQ(viaCandidates, viaResult);
  // Sanity: the fine (reranked) top-3 for [0,1,0,0] is r3 < r2 < r1.
  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(viaCandidates.size(), 3u);
  EXPECT_EQ(std::get<0>(viaCandidates[0]), getId("<r3>").getBits());
  EXPECT_EQ(std::get<0>(viaCandidates[1]), getId("<r2>").getBits());
  EXPECT_EQ(std::get<0>(viaCandidates[2]), getId("<r1>").getBits());
}

// _____________________________________________________________________________
// TWO-LAYER FORM P (pre-filter): the bound candidates are scored with the
// coarse-scan-then-rerank pipeline restricted to the bound set.
// `vec:bindScore` is the FINE distance (bit-equal to `vec:distance` over the
// same rows), `vec:bindCoarseScore` the coarse scan distance -- on the
// designed probe `<r5>` the two differ by the i8 quantization error.
TEST(VectorRerank, formPAnnotateRerankOnBoundSet) {
  auto* qec = qecWithRerankIndexes();
  QueryExecutionTree qet = planQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { ?e <is-a> <RItem> . "
          "SERVICE vec: { _:c vec:index \"embrr\" ; "
          "vec:queryVector \"0,1,0,0\" ; vec:candidates ?e ; vec:result ?e ; "
          "vec:bindScore ?d ; vec:bindCoarseScore ?dc . } "
          "BIND(vec:distance(vidx:embrr, ?e, \"0,1,0,0\") AS ?dref) }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  size_t drefCol = qet.getVariableColumn(Variable{"?dref"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());
  // No `vec:k`: EVERY bound candidate (all six items) is annotated.
  ASSERT_EQ(table.numRows(), 6u);
  bool sawR5 = false;
  for (size_t r = 0; r < table.numRows(); ++r) {
    // The fine score is bit-equal to the exact `vec:distance` (both read the
    // fine rerank layer).
    ASSERT_EQ(table(r, dCol).getDatatype(), Datatype::Double);
    EXPECT_EQ(table(r, dCol), table(r, drefCol)) << "row " << r;
    double fine = table(r, dCol).getDouble();
    double coarse = table(r, dcCol).getDouble();
    if (table(r, eCol) == getId("<r5>")) {
      sawR5 = true;
      // Fine: ~1 - 0.003 (bf16 keeps the small component); coarse: exactly 1
      // (i8 dropped it) -- the quantization error survives the FORM P path.
      EXPECT_NEAR(fine, 0.997, 1e-3);
      EXPECT_NEAR(coarse, 1.0, 1e-6);
      EXPECT_GT(std::abs(fine - coarse), 1e-3);
    } else {
      EXPECT_NEAR(fine, coarse, 0.02) << "row " << r;
    }
  }
  EXPECT_TRUE(sawR5);
}

// _____________________________________________________________________________
// TWO-LAYER FORM P is restricted to the bound set: with only the seed `<r2>`
// bound, the result is `<r2>` alone -- although the whole-index top-1 for the
// query point would be `<r3>`.
TEST(VectorRerank, formPTwoLayerRestrictsToBoundSet) {
  auto* qec = qecWithRerankIndexes();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT * WHERE { ?x <is-a> <SeedItem> . "
               "SERVICE vec: { _:c vec:index \"embrr\" ; "
               "vec:queryVector \"0,1,0,0\" ; vec:candidates ?x ; "
               "vec:result ?nn ; vec:bindScore ?d ; vec:bindCoarseScore ?dc ; "
               "vec:k 3 . } }");
  auto result = qet.getResult();
  size_t nnCol = qet.getVariableColumn(Variable{"?nn"});
  size_t dcCol = qet.getVariableColumn(Variable{"?dc"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(table.numRows(), 1u);
  EXPECT_EQ(table(0, nnCol), getId("<r2>"));
  // The coarse score column is present and a real distance.
  ASSERT_EQ(table(0, dcCol).getDatatype(), Datatype::Double);
}

// _____________________________________________________________________________
// Parse-time validation of the new SERVICE parameters.
TEST(VectorRerank, parseErrors) {
  auto* qec = qecWithRerankIndexes();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
  // Non-positive rerankK.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:rerankK 0 .")),
      HasSubstr("positive integer"));
  // `<bindCoarseScore>` must differ from `<bindScore>` and `<result>`.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; vec:bindScore ?d ; "
                           "vec:bindCoarseScore ?d .")),
      HasSubstr("must be different"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:result ?x ; "
                           "vec:queryVector \"0,1,0,0\" ; "
                           "vec:bindCoarseScore ?x .")),
      HasSubstr("must be different"));
  // `<bindCoarseScore>` needs a query point (it is not available in the
  // entity-to-entity FORM E, which has none).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:candidates ?c ; "
                           "vec:result ?x ; vec:bindCoarseScore ?dc .")),
      HasSubstr("requires a query point"));
  // `<candidates>` and `<result>` must differ WITHOUT a query point (FORM E:
  // a candidate's neighbours are distinct entities; the in-place annotate
  // form requires a query point).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:candidates ?x ; "
                           "vec:result ?x .")),
      HasSubstr("must be different"));
}

namespace {
// A varied bf16 fixture for the `vec:bf16Kernel` cross-check: `KERNEL_ROWS`
// rows of `KERNEL_DIM` (a multiple of 32, so both hand-rolled kernels apply),
// each a distinct pseudo-random unit-ish vector so the cosine distances are
// distinct and the top-k is unambiguous.
constexpr size_t KERNEL_DIM = 128;
constexpr size_t KERNEL_ROWS = 500;

std::vector<float> kernelRow(uint64_t seed, size_t dim = KERNEL_DIM) {
  // splitmix64 for deterministic, well-spread components.
  auto next = [&seed]() {
    uint64_t z = (seed += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
  };
  std::vector<float> v(dim);
  for (float& x : v) {
    x = static_cast<int32_t>(next()) * (1.0f / 2147483648.0f);
  }
  return v;
}

std::string buildKernelFixture(std::string name, VectorScalar scan,
                               std::optional<VectorScalar> rerank,
                               size_t rows = KERNEL_ROWS,
                               size_t dim = KERNEL_DIM) {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = dim;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = scan;
  cfg.rerankScalar_ = rerank;
  cfg.buildHnsw_ = false;
  VectorIndexBuilder builder{basename, cfg};
  for (size_t i = 0; i < rows; ++i) {
    builder.add(mkId(100 + i), "<http://ex/" + std::to_string(i) + ">",
                kernelRow(0x1234'0000ULL + i, dim));
  }
  builder.build();
  return basename;
}
}  // namespace

// The `vec:bf16Kernel` selector: on this CPU (which has AMX + AVX-512-BF16, so
// both hand-rolled kernels run) the SIMD, AMX, punned/GEMM (Auto), and the
// per-row Punned kernels must all return the SAME exact-bf16-cosine top-k, to
// the documented ~1e-6 tolerance -- for the whole-index full sweep, the
// two-layer coarse+rerank scattered pass, and the CSLS full fine sweep. A pure
// performance dial: it never changes results. Bit-identity of DISTANCES is
// required (the finalize is shared); the entity choice at the k-boundary can
// legitimately differ only under an exact tie, which this well-spread fixture
// avoids.
TEST(VectorRerank, bf16KernelSelectorAgreesAcrossKernels) {
  using qlever::vector::Bf16Kernel;
  const std::vector<float> query = kernelRow(0xDEAD'BEEFULL);
  const std::array<Bf16Kernel, 4> kernels = {
      Bf16Kernel::Auto, Bf16Kernel::Amx, Bf16Kernel::Simd, Bf16Kernel::Punned};
  const size_t k = 25;

  auto expectSameTopK = [&](const auto& ref, const auto& other,
                            const char* what, Bf16Kernel kern) {
    ASSERT_EQ(ref.size(), other.size()) << what << " kernel=" << int(kern);
    for (size_t i = 0; i < ref.size(); ++i) {
      // Distance bit-identity (shared finalize) -- and equal entities (no ties
      // in this fixture).
      EXPECT_NEAR(ref[i].distance_, other[i].distance_, 1e-6f)
          << what << " kernel=" << int(kern) << " i=" << i;
      EXPECT_EQ(ref[i].entity_, other[i].entity_)
          << what << " kernel=" << int(kern) << " i=" << i;
    }
  };

  // --- Single-layer bf16 index: the whole-index full fine sweep. ---
  {
    std::string basename =
        buildKernelFixture("k1", VectorScalar::Bf16, std::nullopt);
    VectorIndex idx;
    idx.open(basename, "k1");
    // Reference: the punned per-row metric (kernel-independent baseline).
    auto ref = idx.searchExact(query, k, std::nullopt, std::nullopt, {},
                               nullptr, Bf16Kernel::Punned);
    ASSERT_EQ(ref.size(), k);
    for (Bf16Kernel kern : kernels) {
      auto got = idx.searchExact(query, k, std::nullopt, std::nullopt, {},
                                 nullptr, kern);
      expectSameTopK(ref, got, "single-layer whole-index sweep", kern);
    }
    // CSLS full fine sweep (single-layer): `computeCslsReranked` runs one full
    // CONTIGUOUS fine sweep (the block kernels) and returns the fine distances
    // in coarse order; every kernel must produce the SAME distances. (No csls
    // sidecar needed -- the raw rerank stage, before any cut.)
    auto refR = idx.computeCslsReranked(query, /*neighbors=*/5, /*floorF=*/0.1f,
                                        /*widenF=*/1.0f, /*rerankCap=*/0,
                                        std::nullopt, {}, false,
                                        Bf16Kernel::Punned);
    ASSERT_EQ(refR.fineDist_.size(), KERNEL_ROWS);
    for (Bf16Kernel kern : kernels) {
      auto gotR = idx.computeCslsReranked(query, 5, 0.1f, 1.0f, 0, std::nullopt,
                                          {}, false, kern);
      ASSERT_EQ(gotR.fineDist_.size(), refR.fineDist_.size())
          << "CSLS sweep kernel=" << int(kern);
      ASSERT_EQ(gotR.entityBits_.size(), refR.entityBits_.size());
      for (size_t i = 0; i < refR.fineDist_.size(); ++i) {
        EXPECT_EQ(gotR.entityBits_[i], refR.entityBits_[i])
            << "CSLS sweep kernel=" << int(kern) << " i=" << i;
        EXPECT_NEAR(gotR.fineDist_[i], refR.fineDist_[i], 1e-6f)
            << "CSLS sweep kernel=" << int(kern) << " i=" << i;
      }
    }
    cleanupTmp(basename, "k1");
  }

  // --- Two-layer binary+bf16 index: coarse scan + SCATTERED fine rerank. ---
  // The coarse pass ranks the binary layer; the fine rerank scores those
  // coarse-ranked (i.e. scattered) rows on the bf16 layer -- the SIMD-gather
  // path. `searchExactByRows` takes the kernel and must agree across all.
  {
    std::string basename =
        buildKernelFixture("k2", VectorScalar::Binary, VectorScalar::Bf16);
    VectorIndex idx;
    idx.open(basename, "k2");
    // Coarse-rank a wide margin, then rerank exactly those rows on the fine
    // layer (the production coarse+rerank shape).
    const size_t rerankK = 200;
    auto coarseRows =
        idx.searchExactCoarseWithRows(query, rerankK, std::nullopt);
    ASSERT_FALSE(coarseRows.empty());
    ql::span<const ScoredRow> rowsSpan{coarseRows};
    auto ref = idx.searchExactByRows(query, k, rowsSpan, std::nullopt, {},
                                     Bf16Kernel::Punned);
    ASSERT_EQ(ref.size(), k);
    for (Bf16Kernel kern : kernels) {
      auto got =
          idx.searchExactByRows(query, k, rowsSpan, std::nullopt, {}, kern);
      expectSameTopK(ref, got, "two-layer scattered rerank", kern);
    }
    cleanupTmp(basename, "k2");
  }
}

// _____________________________________________________________________________
// The RESTRICTED (non-covering) candidate search: `searchExact(k, candidates)`
// with a candidate SUBSET takes the scattered-gather branch of the exact
// search, which used to ignore `vec:bf16Kernel` and score through the per-row
// punned metric only -- the ONE rerank path with a different dot engine, so a
// candidate-restricted `?score` could disagree (~1e-6) with the exact
// `vec:distance` of the same pair. Now it routes through the same batched
// SIMD gather as every other rerank path, so all kernels must agree on the
// top-k (distances to ~1e-6, and equal entities -- this well-spread fixture
// has no ties).
TEST(VectorRerank, bf16KernelAgreesOnRestrictedCandidateSearch) {
  using qlever::vector::Bf16Kernel;
  const std::vector<float> query = kernelRow(0xDEAD'BEEFULL);
  const std::array<Bf16Kernel, 4> kernels = {
      Bf16Kernel::Auto, Bf16Kernel::Amx, Bf16Kernel::Simd, Bf16Kernel::Punned};
  const size_t k = 25;
  // Two-layer binary+bf16: `searchExact` scores the FINE bf16 layer, where
  // the hand-rolled kernels apply.
  std::string basename =
      buildKernelFixture("k3", VectorScalar::Binary, VectorScalar::Bf16);
  VectorIndex idx;
  idx.open(basename, "k3");
  // Every third entity: a genuinely NON-COVERING subset, so the search cannot
  // shortcut to the covering whole-index sweep -- it must run the scattered
  // candidate gather under test.
  std::vector<Id> candidates;
  for (size_t i = 0; i < KERNEL_ROWS; i += 3) {
    candidates.push_back(mkId(100 + i));
  }
  std::optional<ql::span<const Id>> candSpan{candidates};
  size_t scoredRef = 0;
  auto ref = idx.searchExact(query, k, candSpan, std::nullopt, {}, &scoredRef,
                             Bf16Kernel::Punned);
  ASSERT_EQ(ref.size(), k);
  // Sanity: only the subset's members were scored (the restricted branch ran).
  ASSERT_EQ(scoredRef, candidates.size());
  for (Bf16Kernel kern : kernels) {
    size_t scored = 0;
    auto got =
        idx.searchExact(query, k, candSpan, std::nullopt, {}, &scored, kern);
    ASSERT_EQ(scored, candidates.size()) << "kernel=" << int(kern);
    ASSERT_EQ(got.size(), ref.size()) << "kernel=" << int(kern);
    for (size_t i = 0; i < ref.size(); ++i) {
      EXPECT_NEAR(ref[i].distance_, got[i].distance_, 1e-6f)
          << "restricted scattered search kernel=" << int(kern) << " i=" << i;
      EXPECT_EQ(ref[i].entity_, got[i].entity_)
          << "restricted scattered search kernel=" << int(kern) << " i=" << i;
    }
  }
  // `maxDistance` filters identically on the batched branch. Cut midway
  // between two consecutive distances (their gap dwarfs the ~1e-6
  // cross-kernel wobble, so the survivor set is unambiguous).
  ASSERT_GT(ref[k / 2 + 1].distance_ - ref[k / 2].distance_, 1e-4f);
  const float maxDist =
      0.5f * (ref[k / 2].distance_ + ref[k / 2 + 1].distance_);
  auto refFiltered = idx.searchExact(query, k, candSpan, maxDist, {}, nullptr,
                                     Bf16Kernel::Punned);
  ASSERT_EQ(refFiltered.size(), k / 2 + 1);
  for (Bf16Kernel kern : kernels) {
    auto gotFiltered =
        idx.searchExact(query, k, candSpan, maxDist, {}, nullptr, kern);
    ASSERT_EQ(gotFiltered.size(), refFiltered.size()) << "kernel=" << int(kern);
  }
  cleanupTmp(basename, "k3");
}

// _____________________________________________________________________________
// The hard `MAX_SEARCH_RESULTS` top-k cap (shrunk via the test seam) applies
// to every plain top-k primitive -- but the annotate form (`keepAll`) is
// exempt on EVERY primitive of its path, because its contract is to score and
// return ALL bound candidates: primitive-level trace over the whole two-layer
// pipeline, then end-to-end through the SERVICE (the annotate FORM P returns
// every candidate even when their count exceeds the cap, while plain top-k
// still caps).
TEST(VectorRerank, annotateFormScoresAllCandidatesBeyondCap) {
  auto* qec = qecWithRerankIndexes();
  auto vidx = getVectorIndex(qec->getIndex(), "embrr");
  ASSERT_TRUE(vidx != nullptr);
  ASSERT_EQ(vidx->numLiveVectors(), 6u);
  const std::vector<float> query{0.f, 1.f, 0.f, 0.f};
  auto getId = makeGetId(qec->getIndex());

  // RAII: shrink the cap below the candidate count, restore on exit.
  struct ScopedCap {
    size_t old_ = qlever::vector::detail::maxSearchResultsForTesting();
    explicit ScopedCap(size_t v) {
      qlever::vector::detail::maxSearchResultsForTesting() = v;
    }
    ~ScopedCap() {
      qlever::vector::detail::maxSearchResultsForTesting() = old_;
    }
  } capGuard{3};
  // Nothing computed under the tiny cap may be served to other tests (nor
  // stale full-cap results to this one).
  qec->getQueryTreeCache().clearAll();

  std::vector<Id> candidates;
  for (const char* e : {"<r0>", "<r1>", "<r2>", "<r3>", "<r4>", "<r5>"}) {
    candidates.push_back(getId(e));
  }
  std::optional<ql::span<const Id>> candSpan{candidates};

  // --- Primitive level: every primitive of the annotate path. ---
  // The exact search (the single-layer / full-precision branch).
  EXPECT_EQ(vidx->searchExact(query, 6, candSpan).size(), 3u);
  EXPECT_EQ(vidx->searchExact(query, 6, candSpan, std::nullopt, {}, nullptr,
                              Bf16Kernel::Auto, /*keepAll=*/true)
                .size(),
            6u);
  // The two-layer pipeline: the coarse pass (the `rerankK` clamp), then the
  // fine rerank over its rows.
  EXPECT_EQ(vidx->searchExactCoarseWithRows(query, 6, candSpan).size(), 3u);
  auto coarseAll = vidx->searchExactCoarseWithRows(
      query, 6, candSpan, std::nullopt, {}, nullptr, /*keepAll=*/true);
  ASSERT_EQ(coarseAll.size(), 6u);
  ql::span<const ScoredRow> rowsSpan{coarseAll};
  EXPECT_EQ(vidx->searchExactByRows(query, 6, rowsSpan).size(), 3u);
  EXPECT_EQ(vidx->searchExactByRows(query, 6, rowsSpan, std::nullopt, {},
                                    Bf16Kernel::Auto, /*keepAll=*/true)
                .size(),
            6u);
  // The by-entity twins (the `vec:query <iri>` flow).
  Id q = getId("<r2>");
  EXPECT_EQ(vidx->searchExactByEntity(q, 6, candSpan).size(), 3u);
  EXPECT_EQ(vidx->searchExactByEntity(q, 6, candSpan, std::nullopt, {},
                                      nullptr, Bf16Kernel::Auto,
                                      /*keepAll=*/true)
                .size(),
            6u);
  EXPECT_EQ(vidx->searchExactCoarseByEntityWithRows(q, 6, candSpan).size(),
            3u);
  auto coarseByEntity = vidx->searchExactCoarseByEntityWithRows(
      q, 6, candSpan, std::nullopt, {}, nullptr, /*keepAll=*/true);
  ASSERT_EQ(coarseByEntity.size(), 6u);
  ql::span<const ScoredRow> rowsByEntity{coarseByEntity};
  EXPECT_EQ(vidx->searchExactByRowsByEntity(q, 6, rowsByEntity).size(), 3u);
  EXPECT_EQ(vidx->searchExactByRowsByEntity(q, 6, rowsByEntity, std::nullopt,
                                            {}, Bf16Kernel::Auto,
                                            /*keepAll=*/true)
                .size(),
            6u);

  // --- End-to-end: the annotate FORM P (no `vec:k`) returns ALL 6 bound
  // candidates although the cap is 3 -- on the two-layer index (the coarse
  // `rerankK` clamp plus the rerank-by-rows clamp) AND the single-layer one
  // (the plain `searchExact` clamp). ---
  for (const char* index : {"embrr", "embbf"}) {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { ?e <is-a> <RItem> . "
                 "SERVICE vec: { _:c vec:index \"" +
                 index +
                 "\" ; vec:queryVector \"0,1,0,0\" ; vec:candidates ?e ; "
                 "vec:result ?e ; vec:bindScore ?dCap . } }");
    auto result = qet.getResult();
    EXPECT_EQ(result->idTable().numRows(), 6u) << index;
  }
  // A genuine top-k (distinct `<result>`, explicit `vec:k 6`) KEEPS the cap:
  // only the 3 nearest come back.
  {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT * WHERE { ?e <is-a> <RItem> . "
                 "SERVICE vec: { _:c vec:index \"embrr\" ; "
                 "vec:queryVector \"0,1,0,0\" ; vec:candidates ?e ; "
                 "vec:result ?nnCap ; vec:k 6 . } }");
    auto result = qet.getResult();
    EXPECT_EQ(result->idTable().numRows(), 3u);
  }
  qec->getQueryTreeCache().clearAll();
}

// _____________________________________________________________________________
// A pinned `vec:bf16Kernel "amx"` must be THREAD-COUNT INVARIANT: the AMX
// block kernel scores full 32-row tile groups on AMX and each block's < 32-row
// remainder with the SIMD one-row dot, whose fp32 accumulation order differs
// (~1 ulp). The per-thread partition boundaries are therefore aligned to the
// 32-row group, so WHICH rows take the SIMD path no longer depends on the
// team size -- the whole-index AMX sweep returns BIT-identical distances for
// every thread count (the serial one-thread sweep is the reference).
TEST(VectorRerank, amxPinnedKernelIsThreadCountInvariant) {
#ifndef _OPENMP
  GTEST_SKIP() << "Without OpenMP there is no partitioning to test.";
#else
  if (!qlever::vector::bf16kernels::amxAvailable()) {
    GTEST_SKIP() << "This CPU has no AMX-BF16.";
  }
  // Above the parallel threshold (2048) so the sweep actually partitions
  // across threads, and NOT a multiple of 32 so the true end exercises the
  // SIMD remainder.
  constexpr size_t N = 4099;
  constexpr size_t DIM = 256;
  std::string basename =
      buildKernelFixture("amxinv", VectorScalar::Bf16, std::nullopt, N, DIM);
  VectorIndex idx;
  idx.open(basename, "amxinv");
  const std::vector<float> query = kernelRow(0xF00D'F00DULL, DIM);
  // RAII: whatever happens, later tests keep the default team size.
  struct OmpGuard {
    int old_ = omp_get_max_threads();
    ~OmpGuard() { omp_set_num_threads(old_); }
  } ompGuard;
  // (distance bits, entity bits) rows -- an exact, printable comparison.
  auto asBits = [](const std::vector<ScoredEntity>& v) {
    std::vector<std::pair<uint32_t, uint64_t>> out;
    out.reserve(v.size());
    for (const auto& e : v) {
      out.emplace_back(absl::bit_cast<uint32_t>(e.distance_),
                       e.entity_.getBits());
    }
    return out;
  };
  // Reference: the serial sweep (k = N returns EVERY row, so any 1-ulp drift
  // anywhere in the index shows up).
  omp_set_num_threads(1);
  auto ref = idx.searchExact(query, N, std::nullopt, std::nullopt, {}, nullptr,
                             Bf16Kernel::Amx);
  ASSERT_EQ(ref.size(), N);
  const auto refBits = asBits(ref);
  for (int t : {2, 3, 5, 7}) {
    omp_set_num_threads(t);
    auto got = idx.searchExact(query, N, std::nullopt, std::nullopt, {},
                               nullptr, Bf16Kernel::Amx);
    EXPECT_EQ(asBits(got), refBits) << "threads=" << t;
  }
  cleanupTmp(basename, "amxinv");
#endif
}
