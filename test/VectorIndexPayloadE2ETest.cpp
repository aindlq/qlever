// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// End-to-end test of the core "index-payload" vector-search flow (see
// `docs/vector-index/index-payload-design.md`): vectors enter QLever as a
// `.npy` bundle (matrix + row-aligned IRI sidecar) through the REGISTERED
// build hook, are loaded through the REGISTERED load hook, and are reached
// from SPARQL only via the `vec:distance` function -- no vector ever becomes
// an RDF term. Ranking is plain `BIND` + `FILTER(BOUND)` + `ORDER BY`;
// entities without a vector get an `UNDEF` distance.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "parser/SparqlParser.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/json.h"

namespace {
using ad_utility::testing::getQec;
using ad_utility::testing::makeGetId;

constexpr std::string_view PREFIX =
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n";

// The test knowledge graph: four entities that get a vector below, plus
// `<nomember>`, which is in the graph but deliberately NOT in the vector
// index. `<is-a> <Item>` is the "natural enumerator" pattern of the design
// doc: the real predicate the embeddings were derived from.
constexpr std::string_view KG =
    "<a> <is-a> <Item> . <b> <is-a> <Item> . <c> <is-a> <Item> . "
    "<d> <is-a> <Item> . <nomember> <is-a> <Item> .";

// The dimension-3 vectors, row i of the `.npy` matrix belonging to line i of
// the IRI sidecar. With the cosine metric and the query point [1,0,0]:
// `<a>` and `<c>` are exact matches (distance ~0), `<b>` is at 1-0.6 = 0.4,
// and `<d>` is orthogonal (distance 1) -- an unambiguous ranking.
const std::vector<std::pair<std::string, std::vector<float>>>& testVectors() {
  static const std::vector<std::pair<std::string, std::vector<float>>> vecs{
      {"<a>", {1.f, 0.f, 0.f}},
      {"<b>", {0.6f, 0.8f, 0.f}},
      {"<c>", {1.f, 0.f, 0.f}},
      {"<d>", {0.f, 0.f, 1.f}},
  };
  return vecs;
}

// Write `rows` as a NumPy v1.0 `.npy` file (little-endian float32, C-order)
// plus the row-aligned IRI sidecar -- the exact input bundle of the design.
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

// The shared test context: build the vector index "emb" from the `.npy`
// bundle via the REGISTERED build hook (the `--service-index` path:
// `parseSpec` -> `buildFromNpy` -> IRI resolution against the KG vocabulary),
// then attach it via the REGISTERED load hook -- the same two hooks `qlever
// index` and the server run. Idempotent (the `getQec` context is cached).
QueryExecutionContext* qecWithPayloadIndex() {
  QueryExecutionContext* qec = getQec(std::string{KG});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, testVectors());
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"emb","npy":")" + npy + R"(","iris":")" +
      iris + R"(","metric":"cosine","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  // The load hook auto-materialized the per-index metadata triples as DELTA
  // triples, but the cached `getQec` context captured its located-triples
  // snapshot at construction -- refresh it so queries see them (the server
  // takes a fresh snapshot per query; the test context does not).
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  // The load hook memory-maps the index, which keeps the data alive; the
  // directory entries (and the input bundle) can go right away.
  for (auto* suffix :
       {".vec.emb.meta", ".vec.emb.keys", ".vec.emb.rowmap", ".vec.emb.data",
        ".vec.emb.iris", ".vec.emb.hnsw", ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
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
// The bf16 fixture: a SEPARATE knowledge graph (a distinct `getQec` cache
// entry) whose vector index is built with `"scalar": "bf16"` from the same
// fp32 `.npy` bundle format. Every vector component is exactly
// bf16-representable (0, 0.25, 0.5, 1), so truncating the fp32 input to the
// 2-byte bf16 store is lossless and the cosine distances are exact.
constexpr std::string_view KG_BF16 =
    "<a> <is-a> <Bf16Item> . <b> <is-a> <Bf16Item> . "
    "<c> <is-a> <Bf16Item> . <d> <is-a> <Bf16Item> .";

// The dimension is deliberately large enough that the page-rounded on-disk
// size of the flat store distinguishes 2-byte bf16 from 4-byte fp32 rows
// (tiny matrices land in a single page either way).
constexpr size_t BF16_DIM = 1024;
constexpr size_t BF16_NUM_VECTORS = 4;

// `<a>` and `<c>` equal the basis vector e0 (distance 0 to an e0 query),
// `<b>` is at 45 degrees in the e0/e1 plane (cosine distance 1 - 1/sqrt(2)),
// and `<d>` is orthogonal (distance 1).
std::vector<std::pair<std::string, std::vector<float>>> bf16TestVectors() {
  std::vector<float> e0(BF16_DIM, 0.f);
  e0[0] = 1.f;
  std::vector<float> diag(BF16_DIM, 0.f);
  diag[0] = 0.5f;
  diag[1] = 0.5f;
  std::vector<float> e2(BF16_DIM, 0.f);
  e2[2] = 1.f;
  return {{"<a>", e0}, {"<b>", diag}, {"<c>", e0}, {"<d>", e2}};
}

// Build the "embbf16" index through the registered hooks and return the qec
// together with the on-disk size of its `.vec.embbf16.data` flat store
// (measured before the fixture unlinks the mmap-ed files).
std::pair<QueryExecutionContext*, std::uintmax_t> qecWithBf16Index() {
  static std::uintmax_t dataFileSize = 0;
  QueryExecutionContext* qec = getQec(std::string{KG_BF16});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return {qec, dataFileSize};
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-bf16-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, bf16TestVectors());
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"embbf16","npy":")" + npy + R"(","iris":")" +
      iris + R"(","metric":"cosine","scalar":"bf16","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  dataFileSize = std::filesystem::file_size(basename + ".vec.embbf16.data");
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  for (auto* suffix :
       {".vec.embbf16.meta", ".vec.embbf16.keys", ".vec.embbf16.rowmap",
        ".vec.embbf16.data", ".vec.embbf16.iris", ".vec.embbf16.hnsw",
        ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
  }
  return {qec, dataFileSize};
}
}  // namespace

// _____________________________________________________________________________
// The flagship query of the design doc: enumerate the entities by their
// natural pattern, BIND the distance to an explicit query vector, drop
// non-members via FILTER(BOUND), and ORDER BY distance -- filtered top-k
// vector search from QLever's own operators, no SERVICE, no HTTP. (The LIMIT
// of the `... LIMIT 3` idiom is applied lazily during export, so the ranking
// is asserted on the ordered table.)
TEST(VectorIndexPayloadE2E, rankEntitiesByDistanceToQueryVector) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                         "BIND(vec:distance(\"emb\", ?e, \"1,0,0\") AS ?d) "
                         "FILTER(BOUND(?d)) } ORDER BY ?d LIMIT 3");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  // `<nomember>` (no vector -> UNDEF) is dropped by FILTER(BOUND(?d)); the
  // four indexed entities remain, ranked by cosine distance to [1,0,0].
  ASSERT_EQ(table.numRows(), 4u);
  // The two exact matches <a> and <c> (distance ~0) rank first, in either
  // order (they are an exact tie).
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<a>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<c>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  // Then <b> (distance 1 - 0.6 = 0.4), then the orthogonal <d> (distance 1).
  EXPECT_EQ(table(2, eCol), getId("<b>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 0.4, 1e-3);
  EXPECT_EQ(table(3, eCol), getId("<d>"));
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
  // The ranking is monotone in the distance.
  EXPECT_LE(table(1, dCol).getDouble(), table(2, dCol).getDouble());
  EXPECT_LT(table(2, dCol).getDouble(), table(3, dCol).getDouble());
}

// _____________________________________________________________________________
// The index itself is a queryable RDF resource (design doc, section "idx:
// metadata triples"): at load time the hook auto-materializes thin metadata
// triples on `<.../vectorSearch/index/emb>` as delta triples, so the space is
// introspectable in plain SPARQL -- dimension, metric, precision, and member
// count, all matching the fixture's `.npy` bundle (4 vectors of dimension 3,
// cosine metric, stored as f32).
TEST(VectorIndexPayloadE2E, indexMetadataTriplesAreQueryable) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet = planQuery(
      qec,
      "SELECT ?dim ?metric ?precision ?count WHERE { "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/emb> "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/dimension> ?dim ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/metric> ?metric ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision> ?precision ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/count> ?count }");
  auto result = qet.getResult();
  const IdTable& table = result->idTable();

  // Exactly one metadata record for the index.
  ASSERT_EQ(table.numRows(), 1u);
  // The metadata literals live in the delta triples' local vocab.
  auto getLiteral = [&](std::string_view varName) {
    Id id = table(0, qet.getVariableColumn(Variable{std::string{varName}}));
    EXPECT_EQ(id.getDatatype(), Datatype::LocalVocabIndex)
        << varName << " should be a (local vocab) literal";
    return id.getLocalVocabIndex()->toStringRepresentation();
  };
  EXPECT_EQ(table(0, qet.getVariableColumn(Variable{"?dim"})),
            Id::makeFromInt(3));
  EXPECT_EQ(getLiteral("?metric"), "\"cosine\"");
  EXPECT_EQ(getLiteral("?precision"), "\"f32\"");
  EXPECT_EQ(table(0, qet.getVariableColumn(Variable{"?count"})),
            Id::makeFromInt(4));
}

// _____________________________________________________________________________
// An entity of the knowledge graph WITHOUT a vector in the index evaluates to
// UNDEF (the design's implicit membership filter), not an error and not a
// number; every indexed entity gets a real double.
TEST(VectorIndexPayloadE2E, nonMemberDistanceIsUndef) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                         "BIND(vec:distance(\"emb\", ?e, \"1,0,0\") AS ?d) }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  // All five entities (including the vectorless `<nomember>`).
  ASSERT_EQ(table.numRows(), 5u);
  bool sawNonMember = false;
  for (size_t r = 0; r < table.numRows(); ++r) {
    if (table(r, eCol) == getId("<nomember>")) {
      EXPECT_TRUE(table(r, dCol).isUndefined());
      sawNonMember = true;
    } else {
      EXPECT_EQ(table(r, dCol).getDatatype(), Datatype::Double)
          << "row " << r << " should have a real distance";
    }
  }
  EXPECT_TRUE(sawNonMember);
}

// _____________________________________________________________________________
// The bf16 storage scalar: the `.npy` input stays fp32 (numpy has no portable
// bf16), but `"scalar": "bf16"` stores every vector as 2-byte bf16. The
// fixture's components are exactly bf16-representable, so the f32 -> bf16
// truncation is lossless and `vec:distance` ranks exactly like an fp32 store:
// identical vectors at distance ~0, orthogonal at cosine distance ~1.
TEST(VectorIndexPayloadE2E, bf16ScalarStoresTwoBytesAndRanksCorrectly) {
  auto [qec, dataFileSize] = qecWithBf16Index();

  // Proof that the flat store holds 2-byte bf16, not 4-byte fp32: the
  // `.vec.embbf16.data` file must fit the bf16 payload plus at most one page
  // of `MmapVector` rounding and its 32-byte trailer -- strictly less than
  // the raw fp32 payload alone would need.
  const std::uintmax_t bf16Payload = BF16_NUM_VECTORS * BF16_DIM * 2;
  EXPECT_GE(dataFileSize, bf16Payload);
  EXPECT_LE(dataFileSize,
            bf16Payload + static_cast<std::uintmax_t>(getpagesize()) + 32);
  EXPECT_LT(dataFileSize, BF16_NUM_VECTORS * BF16_DIM * 4);

  // The persisted metadata agrees: the auto-materialized `precision` triple
  // of the index says "bf16".
  {
    QueryExecutionTree qet = planQuery(
        qec,
        "SELECT ?precision WHERE { "
        "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/embbf16> "
        "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision> "
        "?precision }");
    auto result = qet.getResult();
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 1u);
    Id id = table(0, qet.getVariableColumn(Variable{"?precision"}));
    ASSERT_EQ(id.getDatatype(), Datatype::LocalVocabIndex);
    EXPECT_EQ(id.getLocalVocabIndex()->toStringRepresentation(), "\"bf16\"");
  }

  // Ranking against the query point e0 (all components bf16-exact, so the
  // only imprecision is the bf16 arithmetic of the distance kernel itself).
  std::string queryVector = "1";
  for (size_t i = 1; i < BF16_DIM; ++i) {
    queryVector += ",0";
  }
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Bf16Item> . "
                         "BIND(vec:distance(\"embbf16\", ?e, \"" +
                         queryVector +
                         "\") AS ?d) "
                         "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  ASSERT_EQ(table.numRows(), 4u);
  // The exact matches <a> and <c> (identical bf16 rows -> distance ~0) rank
  // first, in either order.
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<a>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<c>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  // Then <b> at 45 degrees (cosine distance 1 - 1/sqrt(2)), then the
  // orthogonal <d> at distance 1 (small tolerance for the bf16 kernel).
  EXPECT_EQ(table(2, eCol), getId("<b>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 1.0 - 1.0 / std::sqrt(2.0), 1e-2);
  EXPECT_EQ(table(3, eCol), getId("<d>"));
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
}
