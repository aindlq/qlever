// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// End-to-end tests of the vector-search magic SERVICE: registry-based parsing
// (`MagicServiceRegistry` -> `VectorSearchQuery`), planning
// (`MagicServicePlannerRegistry` -> `planMagicService`), and the
// `VectorSearch`/`VectorSearchJoin` operations, executed against a small test
// knowledge graph with an attached vector index.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>

#include "./engine/ValuesForTesting.h"
#include "./util/GTestHelpers.h"
#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "index/IndexImpl.h"
#include "parser/SparqlParser.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorSearch.h"
#include "services/vectorSearch/VectorSearchJoin.h"

namespace {
using ad_utility::testing::getQec;
using ad_utility::testing::makeGetId;
using ::testing::HasSubstr;

constexpr std::string_view PREFIX =
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n";

// The test knowledge graph: three statues and two paintings. `<e4>`
// deliberately gets NO vector below.
QueryExecutionContext* qecWithKg() {
  return getQec(
      "<e0> <is-a> <Statue> . <e1> <is-a> <Statue> . <e2> <is-a> <Painting> . "
      "<e3> <is-a> <Statue> . <e4> <is-a> <Painting> .");
}

// Dimension-4 unit vectors with an unambiguous cosine neighbourhood: `<e1>` is
// closest to `<e0>`, everything else is orthogonal to `<e0>`.
const std::vector<std::pair<std::string, std::vector<float>>>& testVectors() {
  static const std::vector<std::pair<std::string, std::vector<float>>> vecs{
      {"<e0>", {1.f, 0.f, 0.f, 0.f}},
      {"<e1>", {0.9f, 0.1f, 0.f, 0.f}},
      {"<e2>", {0.f, 1.f, 0.f, 0.f}},
      {"<e3>", {0.f, 0.f, 1.f, 0.f}},
  };
  return vecs;
}

// Build a vector index over the test KG's entities and attach it to the
// (shared, cached) test index as the "vectorSearch" extension. Idempotent.
QueryExecutionContext* qecWithVectorIndex() {
  QueryExecutionContext* qec = qecWithKg();
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  auto getId = makeGetId(qec->getIndex());
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecservicetest-" + std::to_string(::getpid())))
          .string();
  qlever::vector::VectorIndexConfig cfg;
  cfg.name_ = "clip";
  cfg.dimensions_ = 4;
  cfg.metric_ = qlever::vector::VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  qlever::vector::VectorIndexBuilder builder{basename, cfg};
  for (const auto& [iri, vec] : testVectors()) {
    builder.add(getId(iri), vec);
  }
  builder.build();
  qlever::vector::VectorIndex idx;
  idx.open(basename, "clip");
  // The mmap keeps the data alive; the directory entries can go right away.
  for (auto* suffix : {".meta", ".keys", ".data"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.clip" + suffix, ec);
  }
  auto collection = std::make_shared<qlever::vector::VectorIndexCollection>();
  collection->add("clip", std::move(idx));
  impl.setExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME},
                    std::move(collection));
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

// Convenience: run `query` and return (result, column of `var`).
std::pair<std::shared_ptr<const Result>, size_t> runQuery(
    QueryExecutionContext* qec, std::string query, const Variable& var) {
  QueryExecutionTree qet = planQuery(qec, std::move(query));
  size_t col = qet.getVariableColumn(var);
  return {qet.getResult(), col};
}

// A `ValuesForTesting` child binding the given `variables` (nullopt = unnamed
// column) over the single-row table of entity ids `rowIds`.
std::shared_ptr<QueryExecutionTree> makeChild(
    QueryExecutionContext* qec, std::vector<std::optional<Variable>> variables,
    const std::vector<std::vector<Id>>& rows) {
  IdTable table{variables.size(), qec->getAllocator()};
  for (const auto& row : rows) {
    table.emplace_back();
    for (size_t c = 0; c < row.size(); ++c) {
      table(table.numRows() - 1, c) = row[c];
    }
  }
  return ad_utility::makeExecutionTree<ValuesForTesting>(qec, std::move(table),
                                                         std::move(variables));
}
}  // namespace

// _____________________________________________________________________________
TEST(VectorSearchService, wholeIndexQueryVectorForm) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] = runQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
          "vec:queryVector \"1,0,0,0\" ; vec:result ?nn ; vec:bindScore ?d ; "
          "vec:k 2 . } }",
      Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 2u);
  // Nearest is <e0> itself (distance 0), then <e1>.
  EXPECT_EQ(table(0, col), getId("<e0>"));
  EXPECT_EQ(table(1, col), getId("<e1>"));
}

// _____________________________________________________________________________
TEST(VectorSearchService, queryEntityForm) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:query <e0> ; vec:result ?nn ; vec:k 2 . } }",
               Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 2u);
  EXPECT_EQ(table(0, col), getId("<e0>"));
  EXPECT_EQ(table(1, col), getId("<e1>"));
}

// _____________________________________________________________________________
TEST(VectorSearchService, queryEntityWithoutVectorYieldsEmptyResult) {
  auto* qec = qecWithVectorIndex();
  // <e4> exists in the KG but has no vector.
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:query <e4> ; vec:result ?nn ; vec:k 2 . } }",
               Variable{"?nn"});
  EXPECT_EQ(result->idTable().numRows(), 0u);
}

// _____________________________________________________________________________
TEST(VectorSearchService, candidateRestrictedForm) {
  auto* qec = qecWithVectorIndex();
  // Restrict the search space to the paintings <e2> (has a vector) and <e4>
  // (has none): only <e2> can be returned, regardless of better matches
  // elsewhere in the index.
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:queryVector \"1,0,0,0\" ; vec:result ?p ; vec:k 5 . "
                   "{ ?p <is-a> <Painting> } } }",
               Variable{"?p"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 1u);
  EXPECT_EQ(table(0, col), getId("<e2>"));
}

// _____________________________________________________________________________
// Regression test: a candidate pattern that matches NOTHING must produce an
// empty result -- not fall back to searching the whole index.
TEST(VectorSearchService, emptyCandidateSetYieldsEmptyResult) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:queryVector \"1,0,0,0\" ; vec:result ?p ; vec:k 5 . "
                   "{ ?p <is-a> <Vase> } } }",
               Variable{"?p"});
  EXPECT_EQ(result->idTable().numRows(), 0u);
}

// _____________________________________________________________________________
TEST(VectorSearchService, joinForEachForm) {
  auto* qec = qecWithVectorIndex();
  // For each painting ?x, the 2 nearest entities. <e2> has a vector (self +
  // <e1> as its neighbours); <e4> has none and contributes no rows.
  auto [result, col] = runQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
          "vec:left ?x ; vec:result ?nn ; vec:bindScore ?d ; vec:k 2 . "
          "{ ?x <is-a> <Painting> } } }",
      Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 2u);
  EXPECT_EQ(table(0, col), getId("<e2>"));  // self
  EXPECT_EQ(table(1, col), getId("<e1>"));  // nearest other
}

// _____________________________________________________________________________
TEST(VectorSearchService, maxDistanceFilters) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:queryVector \"1,0,0,0\" ; vec:result ?nn ; vec:k 5 ; "
                   "vec:maxDistance 0.0001 . } }",
               Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  // Only <e0> is within distance ~0 of the query.
  ASSERT_EQ(table.numRows(), 1u);
  EXPECT_EQ(table(0, col), getId("<e0>"));
}

// _____________________________________________________________________________
TEST(VectorSearchService, parseErrors) {
  auto* qec = qecWithVectorIndex();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
  // Unknown parameter.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:result ?x ; "
                           "vec:queryVector \"1,0,0,0\" ; vec:frobnicate 3 .")),
      HasSubstr("Unsupported parameter"));
  // Missing index.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                query("_:c vec:queryVector \"1,0,0,0\" ; vec:result ?x .")),
      HasSubstr("requires the `<index>` parameter"));
  // Missing result variable.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(
          qec, query("_:c vec:index \"clip\" ; vec:queryVector \"1,0,0,0\" .")),
      HasSubstr("`<result>`"));
  // More than one query point.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:result ?x ; "
                           "vec:queryVector \"1,0,0,0\" ; vec:query <e0> .")),
      HasSubstr("exactly one of"));
  // Malformed query vector.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:result ?x ; "
                           "vec:queryVector \"1,zwei,3\" .")),
      HasSubstr("not a number"));
  // Empty query vector.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:result ?x ; "
                           "vec:queryVector \"\" .")),
      HasSubstr("at least one number"));
  // Non-positive k.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:result ?x ; "
                           "vec:queryVector \"1,0,0,0\" ; vec:k 0 .")),
      HasSubstr("positive integer"));
  // `<left>` and `<result>` must differ.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:left ?x ; "
                           "vec:result ?x . { ?x <is-a> <Statue> }")),
      HasSubstr("must be different"));
  // The `<left>` form needs a nested pattern.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:left ?x ; "
                           "vec:result ?nn .")),
      HasSubstr("nested"));
}

// _____________________________________________________________________________
TEST(VectorSearchService, planAndExecutionErrors) {
  auto* qec = qecWithVectorIndex();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
  // The join form rejects a nested pattern that also binds the result
  // variable (silently relabeling a child column would be wrong results).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:left ?x ; "
                           "vec:result ?y . { ?x <is-a> ?y }")),
      HasSubstr("must not be bound by the nested query pattern"));
  // The candidate form rejects extra visible variables in the nested pattern.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:queryVector "
                           "\"1,0,0,0\" ; vec:result ?p . { ?p <is-a> ?t }")),
      HasSubstr("must bind only"));
  // Unknown index name: fails at execution time with a clear message.
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               query("_:c vec:index \"nonexistent\" ; vec:queryVector "
                     "\"1,0,0,0\" ; vec:result ?x ."),
               Variable{"?x"}),
      HasSubstr("no loaded vector index named 'nonexistent'"));
  // Dimension mismatch: clear message.
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               query("_:c vec:index \"clip\" ; vec:queryVector \"1,0\" ; "
                     "vec:result ?x ."),
               Variable{"?x"}),
      HasSubstr("dimension"));
  // HNSW requested but the index has none.
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               query("_:c vec:index \"clip\" ; vec:queryVector \"1,0,0,0\" ; "
                     "vec:result ?x ; vec:algorithm vec:hnsw ."),
               Variable{"?x"}),
      HasSubstr("no HNSW structure"));
}

// _____________________________________________________________________________
// Regression test: the cache key of the candidate-restricted form must include
// the COLUMN the result variable is bound to -- two children with identical
// cache keys can bind it to different columns.
TEST(VectorSearchService, cacheKeyIncludesCandidateColumn) {
  auto* qec = qecWithVectorIndex();
  auto getId = makeGetId(qec->getIndex());
  Id e0 = getId("<e0>");
  Id e2 = getId("<e2>");
  qlever::vector::VectorSearchConfiguration config;
  config.indexName_ = "clip";
  config.queryVector_ = std::vector<float>{1.f, 0.f, 0.f, 0.f};
  config.resultVariable_ = Variable{"?p"};
  // Same table content; the result variable ?p sits in column 0 vs. column 1.
  auto childCol0 = makeChild(qec, {Variable{"?p"}, std::nullopt}, {{e2, e0}});
  auto childCol1 = makeChild(qec, {std::nullopt, Variable{"?p"}}, {{e2, e0}});
  VectorSearch opCol0{qec, config, childCol0};
  VectorSearch opCol1{qec, config, childCol1};
  EXPECT_NE(opCol0.getCacheKey(), opCol1.getCacheKey());
  // Clones share the cache key.
  EXPECT_EQ(opCol0.getCacheKey(), opCol0.clone()->getCacheKey());
}

// _____________________________________________________________________________
// Regression test: query vectors that agree to six significant digits must not
// collide on one cache key (`absl::StrCat(float)` rounds to six digits).
TEST(VectorSearchService, cacheKeyIsBitExactInQueryVector) {
  auto* qec = qecWithVectorIndex();
  qlever::vector::VectorSearchConfiguration a;
  a.indexName_ = "clip";
  a.resultVariable_ = Variable{"?x"};
  a.queryVector_ = std::vector<float>{0.10000001f, 0.f, 0.f, 0.f};
  qlever::vector::VectorSearchConfiguration b = a;
  b.queryVector_ = std::vector<float>{0.10000002f, 0.f, 0.f, 0.f};
  VectorSearch opA{qec, a};
  VectorSearch opB{qec, b};
  EXPECT_NE(opA.getCacheKey(), opB.getCacheKey());
}

// _____________________________________________________________________________
// Text and image query points depend on an external embedding endpoint at
// compute time, so their results must not be cached.
TEST(VectorSearchService, embeddingResultsAreNotCacheable) {
  auto* qec = qecWithVectorIndex();
  qlever::vector::VectorSearchConfiguration byVector;
  byVector.indexName_ = "clip";
  byVector.resultVariable_ = Variable{"?x"};
  byVector.queryVector_ = std::vector<float>{1.f, 0.f, 0.f, 0.f};
  EXPECT_TRUE(VectorSearch(qec, byVector).canResultBeCached());

  qlever::vector::VectorSearchConfiguration byText = byVector;
  byText.queryVector_ = std::nullopt;
  byText.queryText_ = "a green statue";
  EXPECT_FALSE(VectorSearch(qec, byText).canResultBeCached());
}

// _____________________________________________________________________________
TEST(VectorSearchService, joinFormEstimatesAndMemoization) {
  auto* qec = qecWithVectorIndex();
  auto getId = makeGetId(qec->getIndex());
  Id e2 = getId("<e2>");
  qlever::vector::VectorSearchConfiguration config;
  config.indexName_ = "clip";
  config.leftVariable_ = Variable{"?x"};
  config.resultVariable_ = Variable{"?nn"};
  config.k_ = 2;
  // Duplicate left values: each row still produces k result rows.
  auto child = makeChild(qec, {Variable{"?x"}}, {{e2}, {e2}, {e2}});
  VectorSearchJoin join{qec, config, child};
  EXPECT_EQ(join.getSizeEstimate(), 6u);
  auto result = join.getResult();
  EXPECT_EQ(result->idTable().numRows(), 6u);
}
