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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
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

// The shared test context: THREE indices over the same vectors, built through
// the REGISTERED build/load hooks (the `--service-index` path incl. the
// `rerank` key of `parseSpec`):
//   * "embrr"  -- the two-layer index under test (scan i8 + rerank bf16);
//   * "embbf"  -- a single-layer bf16 reference (the fine layer's twin);
//   * "embi8"  -- a single-layer i8 reference (the coarse layer's twin).
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
      {"vectorSearch", nlohmann::json::array({entry("embrr", "i8", "bf16"),
                                              entry("embbf", "bf16", nullptr),
                                              entry("embi8", "i8", nullptr)})}};
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  // The two-layer build must produce the rerank matrix ONLY for "embrr".
  EXPECT_TRUE(std::filesystem::exists(vectorRerankDataFile(basename, "embrr")));
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
  for (std::string_view name : {"embrr", "embbf", "embi8"}) {
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
                             std::string name = "rr") {
  std::string basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = name;
  cfg.dimensions_ = BIG_DIM;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.scalar_ = VectorScalar::I8;
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
  // `<bindCoarseScore>` is a whole-index (produce path) feature.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:candidates ?c ; "
                           "vec:result ?x ; vec:bindCoarseScore ?dc .")),
      HasSubstr("only supported"));
  // `<candidates>` and `<result>` must differ (the in-place annotate form is
  // not supported yet).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"embrr\" ; vec:candidates ?x ; "
                           "vec:result ?x .")),
      HasSubstr("must be different"));
}
