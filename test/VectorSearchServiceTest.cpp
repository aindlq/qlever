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
#include <set>

#include "./engine/ValuesForTesting.h"
#include "./util/GTestHelpers.h"
#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "index/IndexExtension.h"
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
    builder.add(getId(iri), iri, vec);
  }
  builder.build();
  qlever::vector::VectorIndex idx;
  idx.open(basename, "clip");
  // The mmap keeps the data alive; the directory entries can go right away.
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
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
// The search SERVICE is a pure "search" surface: a nested `{ ... }` pattern is
// no longer supported (rank an existing set with `vec:distance` instead).
TEST(VectorSearchService, nestedPatternRejected) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                std::string{PREFIX} +
                    "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                    "vec:queryVector \"1,0,0,0\" ; vec:result ?p ; vec:k 5 . "
                    "{ ?p <is-a> <Painting> } } }"),
      HasSubstr("nested"));
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
// The `vec:distance(...)` SPARQL function: BIND it over a bound `?entity` and
// the per-row distance must equal what a `DistanceComputer` (and hence
// `searchExact`) reports for that entity + query point.
TEST(VectorSearchService, vecDistanceBindMatchesComputer) {
  auto* qec = qecWithVectorIndex();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <Statue> . "
               "BIND(vec:distance(\"clip\", ?e, \"1,0,0,0\") AS ?d) }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  // The three statues <e0>, <e1>, <e3> all have vectors.
  ASSERT_EQ(table.numRows(), 3u);

  auto vidx = qlever::vector::getVectorIndex(qec->getIndex(), "clip");
  ASSERT_TRUE(vidx != nullptr);
  auto computer =
      vidx->makeDistanceComputer(std::vector<float>{1.f, 0.f, 0.f, 0.f});
  for (size_t r = 0; r < table.numRows(); ++r) {
    Id entity = table(r, eCol);
    Id distance = table(r, dCol);
    ASSERT_EQ(distance.getDatatype(), Datatype::Double);
    EXPECT_FLOAT_EQ(static_cast<float>(distance.getDouble()), computer(entity))
        << "row " << r;
  }
}

// _____________________________________________________________________________
// BIND(vec:distance) + ORDER BY + LIMIT IS a filtered top-k search using
// QLever's own operators. For the (positive) cosine distances the order must be
// <e0> (self, distance 0) then <e1>.
TEST(VectorSearchService, vecDistanceOrderByLimitIsFilteredSearch) {
  auto* qec = qecWithVectorIndex();
  // ORDER BY the (positive) cosine distances ascending: <e0> (self, 0), then
  // <e1>, then <e3>. (`LIMIT` is applied lazily during result export, so the
  // materialised table keeps all rows; the ordering is what matters here.)
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <Statue> . "
               "BIND(vec:distance(\"clip\", ?e, \"1,0,0,0\") AS ?d) } "
               "ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(table.numRows(), 3u);
  EXPECT_EQ(table(0, eCol), getId("<e0>"));
  EXPECT_EQ(table(1, eCol), getId("<e1>"));
  EXPECT_EQ(table(2, eCol), getId("<e3>"));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_LT(table(0, dCol).getDouble(), table(1, dCol).getDouble());
  EXPECT_LT(table(1, dCol).getDouble(), table(2, dCol).getDouble());
}

// _____________________________________________________________________________
// The query point may be a constant entity IRI whose stored vector is used.
TEST(VectorSearchService, vecDistanceQueryByEntityIri) {
  auto* qec = qecWithVectorIndex();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <Statue> . "
               "BIND(vec:distance(\"clip\", ?e, <e0>) AS ?d) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());
  ASSERT_EQ(table.numRows(), 3u);
  // <e0> is its own nearest neighbour (distance 0).
  EXPECT_EQ(table(0, eCol), getId("<e0>"));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
}

// _____________________________________________________________________________
// An entity without a vector evaluates to UNDEF (and is dropped by ORDER BY /
// comparisons), rather than erroring.
TEST(VectorSearchService, vecDistanceMissingVectorIsUndef) {
  auto* qec = qecWithVectorIndex();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <Painting> . "
               "BIND(vec:distance(\"clip\", ?e, \"1,0,0,0\") AS ?d) }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());
  // <e2> (has a vector) and <e4> (no vector).
  ASSERT_EQ(table.numRows(), 2u);
  for (size_t r = 0; r < table.numRows(); ++r) {
    if (table(r, eCol) == getId("<e2>")) {
      EXPECT_EQ(table(r, dCol).getDatatype(), Datatype::Double);
    } else {
      EXPECT_EQ(table(r, eCol), getId("<e4>"));
      EXPECT_TRUE(table(r, dCol).isUndefined());
    }
  }
}

// _____________________________________________________________________________
// The `ORDER BY ?d LIMIT k` idiom needs `FILTER(BOUND(?d))` when the candidate
// set contains vectorless entities: UNDEF sorts BEFORE every real distance, so
// without the filter the vectorless entity wins the top-1.
TEST(VectorSearchService, vecDistanceOrderByNeedsBoundFilter) {
  auto* qec = qecWithVectorIndex();
  auto getId = makeGetId(qec->getIndex());
  // Paintings = {<e2> (has a vector), <e4> (no vector)}.
  auto topOne = [&](bool withFilter) {
    QueryExecutionTree qet = planQuery(
        qec, std::string{PREFIX} +
                 "SELECT ?e WHERE { ?e <is-a> <Painting> . "
                 "BIND(vec:distance(\"clip\", ?e, \"0,1,0,0\") AS ?d) " +
                 (withFilter ? "FILTER(BOUND(?d)) " : "") +
                 "} ORDER BY ?d LIMIT 1");
    auto result = qet.getResult();
    size_t eCol = qet.getVariableColumn(Variable{"?e"});
    return result->idTable()(0, eCol);
  };
  // Without the filter, the vectorless <e4> (UNDEF distance) sorts first.
  EXPECT_EQ(topOne(false), getId("<e4>"));
  // With FILTER(BOUND(?d)), the true nearest painting <e2> is returned.
  EXPECT_EQ(topOne(true), getId("<e2>"));
}

// _____________________________________________________________________________
// Configuration/usage errors of `vec:distance` surface clearly: parse-time for
// a wrong arity or argument kind, evaluation-time for an unknown index or a
// dimension mismatch.
TEST(VectorSearchService, vecDistanceErrors) {
  auto* qec = qecWithVectorIndex();
  auto selectBind = [](std::string_view call) {
    return std::string{PREFIX} +
           "SELECT ?e ?d WHERE { ?e <is-a> <Statue> . "
           "BIND(" +
           std::string{call} + " AS ?d) }";
  };
  // Wrong arity (parse time).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, selectBind("vec:distance(\"clip\", ?e)")),
      HasSubstr("three arguments"));
  // First argument not a string literal (parse time).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, selectBind("vec:distance(?e, ?e, \"1,0,0,0\")")),
      HasSubstr("string literal naming the vector index"));
  // Third argument neither a float-list literal nor an IRI (parse time).
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, selectBind("vec:distance(\"clip\", ?e, ?e)")),
      HasSubstr("third argument"));
  // Unknown index (evaluation time).
  {
    QueryExecutionTree qet =
        planQuery(qec, selectBind("vec:distance(\"nope\", ?e, \"1,0,0,0\")"));
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(),
                                 HasSubstr("no vector index named"));
  }
  // Dimension mismatch (evaluation time).
  {
    QueryExecutionTree qet =
        planQuery(qec, selectBind("vec:distance(\"clip\", ?e, \"1,0\")"));
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(), HasSubstr("dimension"));
  }
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
      HasSubstr("not a finite number"));
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
      planQuery(
          qec, query("_:c vec:index \"clip\" ; vec:left ?x ; vec:result ?x .")),
      HasSubstr("must be different"));
  // `<left>` and `<bindScore>` must differ.
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, query("_:c vec:index \"clip\" ; vec:left ?x ; "
                           "vec:result ?nn ; vec:bindScore ?x .")),
      HasSubstr("must be different"));
}

// _____________________________________________________________________________
TEST(VectorSearchService, planAndExecutionErrors) {
  auto* qec = qecWithVectorIndex();
  auto query = [](std::string_view body) {
    return std::string{PREFIX} + "SELECT * WHERE { SERVICE vec: { " +
           std::string{body} + " } }";
  };
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
// Every query form is cacheable, including the embedding-based ones: the query
// text/image is part of the cache key, and the endpoint + model are fixed for
// the run, so caching the (expensive-to-embed) result of an identical repeated
// query is correct. The in-memory cache is cleared on restart.
TEST(VectorSearchService, allQueryFormsAreCacheable) {
  auto* qec = qecWithVectorIndex();
  qlever::vector::VectorSearchConfiguration byVector;
  byVector.indexName_ = "clip";
  byVector.resultVariable_ = Variable{"?x"};
  byVector.queryVector_ = std::vector<float>{1.f, 0.f, 0.f, 0.f};
  EXPECT_TRUE(VectorSearch(qec, byVector).canResultBeCached());

  qlever::vector::VectorSearchConfiguration byText = byVector;
  byText.queryVector_ = std::nullopt;
  byText.queryText_ = "a green statue";
  EXPECT_TRUE(VectorSearch(qec, byText).canResultBeCached());
  // The query text is part of the cache key, so distinct texts do not collide.
  qlever::vector::VectorSearchConfiguration otherText = byText;
  otherText.queryText_ = "a red painting";
  EXPECT_NE(VectorSearch(qec, byText).getCacheKey(),
            VectorSearch(qec, otherText).getCacheKey());
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

// _____________________________________________________________________________
// The `<left>` variable can be bound by the SURROUNDING query instead of a
// nested pattern: the operation is planned as an incomplete leaf and completed
// by the join enumeration (the generic `IncompleteJoinOperation` mechanism).
TEST(VectorSearchService, outerBoundLeftJoinForm) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] =
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { ?x <is-a> <Painting> . "
                   "SERVICE vec: { _:c vec:index \"clip\" ; vec:left ?x ; "
                   "vec:result ?nn ; vec:bindScore ?d ; vec:k 2 . } }",
               Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  // <e2> is the only painting with a vector (self + <e1>); <e4> has none.
  ASSERT_EQ(table.numRows(), 2u);
  std::set<uint64_t> nns{table(0, col).getBits(), table(1, col).getBits()};
  EXPECT_TRUE(nns.contains(getId("<e2>").getBits()));
  EXPECT_TRUE(nns.contains(getId("<e1>").getBits()));
}

// _____________________________________________________________________________
// The result variable may ALSO be constrained by the surrounding query (a join
// on the vector-search output). Planning must not abort, and the result is the
// k nearest that also satisfy the constraint. (Regression: an incomplete op
// exposing its output columns must not turn an output-variable connection into
// a normal join that aborts planning.)
TEST(VectorSearchService, outerBoundLeftResultAlsoConstrained) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] = runQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { ?x <is-a> <Painting> . ?nn <is-a> <Statue> . "
          "SERVICE vec: { _:c vec:index \"clip\" ; vec:left ?x ; "
          "vec:result ?nn ; vec:k 5 . } }",
      Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  // <e2>'s neighbours are {e2,e1,e0,e3}; intersect with statues {e0,e1,e3}.
  std::set<uint64_t> got;
  for (size_t i = 0; i < table.numRows(); ++i)
    got.insert(table(i, col).getBits());
  EXPECT_FALSE(got.contains(getId("<e2>").getBits()));  // e2 is a painting
  for (uint64_t id : got) {
    EXPECT_TRUE(id == getId("<e0>").getBits() ||
                id == getId("<e1>").getBits() || id == getId("<e3>").getBits());
  }
  EXPECT_TRUE(got.contains(getId("<e1>").getBits()));  // nearest statue
}

// _____________________________________________________________________________
// A FILTER on the score of an outer-bound-left search must apply after the
// join completes, not attach to the incomplete operation (which would leave it
// unjoinable and fail at runtime).
TEST(VectorSearchService, outerBoundLeftFilterOnScore) {
  auto* qec = qecWithVectorIndex();
  auto [result, col] = runQuery(
      qec,
      std::string{PREFIX} +
          "SELECT * WHERE { ?x <is-a> <Painting> . "
          "SERVICE vec: { _:c vec:index \"clip\" ; vec:left ?x ; "
          "vec:result ?nn ; vec:bindScore ?s ; vec:k 5 . } FILTER(?s < 0.01) }",
      Variable{"?nn"});
  auto getId = makeGetId(qec->getIndex());
  const IdTable& table = result->idTable();
  // Only the self-match (distance ~0) survives the filter.
  ASSERT_EQ(table.numRows(), 1u);
  EXPECT_EQ(table(0, col), getId("<e2>"));
}

// _____________________________________________________________________________
// Taking `<left>` from an OPTIONAL/MINUS pattern is not supported and must be
// a clear error (not silent inner-join / expansion semantics).
TEST(VectorSearchService, outerBoundLeftOptionalMinusRejected) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, std::string{PREFIX} +
                         "SELECT * WHERE { ?x <is-a> <Painting> . OPTIONAL { "
                         "SERVICE vec: { _:c vec:index \"clip\" ; vec:left "
                         "?x ; vec:result ?nn ; vec:k 2 . } } }"),
      HasSubstr("OPTIONAL or MINUS"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, std::string{PREFIX} +
                         "SELECT * WHERE { ?x <is-a> <Painting> . MINUS { "
                         "SERVICE vec: { _:c vec:index \"clip\" ; vec:left "
                         "?x ; vec:result ?nn ; vec:k 2 . } } }"),
      HasSubstr("OPTIONAL or MINUS"));
}

// _____________________________________________________________________________
// Without any binding of `<left>` the operation stays incomplete and fails at
// execution time with an actionable message (not a crash or silent result).
TEST(VectorSearchService, unjoinedOuterLeftThrows) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT * WHERE { SERVICE vec: { _:c vec:index \"clip\" ; "
                   "vec:left ?x ; vec:result ?nn ; vec:k 2 . } }",
               Variable{"?nn"}),
      HasSubstr("is not bound anywhere"));
}

// _____________________________________________________________________________
// `vec:distanceText` embeds the query text at query time via the index's OWN
// endpoint (looked up by the index name). The test "clip" index has no endpoint
// configured, so it fails with the clear "no embedding endpoint" error -- which
// exercises the index-keyed embed path of the function.
TEST(VectorSearchService, vecDistanceTextEmbedsViaIndexEndpoint) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT ?s ?d WHERE { ?s <is-a> <Statue> . "
                   "BIND(vec:distanceText(\"clip\", ?s, \"a green statue\") AS "
                   "?d) } ORDER BY ?d",
               Variable{"?s"}),
      HasSubstr("embedding endpoint"));
}

// _____________________________________________________________________________
// `vec:distanceImage` likewise embeds an image URL via the index's endpoint.
TEST(VectorSearchService, vecDistanceImageEmbedsViaIndexEndpoint) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      runQuery(qec,
               std::string{PREFIX} +
                   "SELECT ?s ?d WHERE { ?s <is-a> <Statue> . "
                   "BIND(vec:distanceImage(\"clip\", ?s, "
                   "<https://example.org/cat.jpg>) AS ?d) } ORDER BY ?d",
               Variable{"?s"}),
      HasSubstr("embedding endpoint"));
}

// _____________________________________________________________________________
// After the RDF data is re-indexed, the vector index is REMAPPED instead of
// rebuilt: `.iris` is re-resolved against the new vocabulary and only the two
// small mapping files are rewritten; the vectors and the HNSW graph (which is
// keyed by row indices, not entity ids) are reused as-is. Entities that
// disappeared become tombstones and vanish from all search results.
TEST(VectorSearchService, remapAfterKgReindex) {
  // "Old" and "new" knowledge graphs: the new one adds <a-new-entity> (which
  // shifts every vocabulary position) and drops <e3> entirely.
  auto* qecOld = getQec(
      "<e0> <is-a> <Statue> . <e1> <is-a> <Statue> . <e2> <is-a> <Painting> . "
      "<e3> <is-a> <Statue> .");
  auto* qecNew = getQec(
      "<a-new-entity> <is-a> <Statue> . <e0> <is-a> <Statue> . "
      "<e1> <is-a> <Statue> . <e2> <is-a> <Painting> .");

  std::string basename = (std::filesystem::temp_directory_path() /
                          ("qlever-vecremaptest-" + std::to_string(::getpid())))
                             .string();
  auto getIdOld = makeGetId(qecOld->getIndex());
  qlever::vector::VectorIndexConfig cfg;
  cfg.name_ = "clip";
  cfg.dimensions_ = 4;
  cfg.metric_ = qlever::vector::VectorMetric::Cosine;
  cfg.buildHnsw_ = true;
  cfg.hnswExpansionSearch_ = 64;
  {
    qlever::vector::VectorIndexBuilder builder{basename, cfg};
    builder.setVocabSize(qecOld->getIndex().getImpl().getVocab().size());
    for (const auto& [iri, vec] : testVectors()) {
      builder.add(getIdOld(iri), iri, vec);
    }
    builder.build();
  }

  // Remap against the NEW knowledge graph.
  auto [live, tombstones] =
      qlever::vector::remapVectorIndex(qecNew->getIndex(), basename, "clip");
  EXPECT_EQ(live, 3u);
  EXPECT_EQ(tombstones, 1u);  // <e3> is gone

  qlever::vector::VectorIndex idx;
  idx.open(basename, "clip");
  EXPECT_EQ(idx.numVectors(), 4u);
  EXPECT_EQ(idx.numLiveVectors(), 3u);
  EXPECT_EQ(idx.metadata().vocabSize_,
            qecNew->getIndex().getImpl().getVocab().size());

  // Lookups now work with the NEW ids, and the vectors are unchanged.
  auto getIdNew = makeGetId(qecNew->getIndex());
  auto v = idx.getVector(getIdNew("<e2>"));
  ASSERT_TRUE(v.has_value());
  EXPECT_FLOAT_EQ((*v)[1], 1.f);

  // Exact search returns the three surviving entities with new ids.
  std::vector<float> queryE0{1.f, 0.f, 0.f, 0.f};
  auto exact = idx.searchExact(queryE0, 10);
  ASSERT_EQ(exact.size(), 3u);
  EXPECT_EQ(exact[0].entity_, getIdNew("<e0>"));
  EXPECT_EQ(exact[1].entity_, getIdNew("<e1>"));

  // The (untouched, row-keyed) HNSW graph also serves the new ids and skips
  // the tombstone, even when more results are requested than remain live.
  // (Note that <e3> cannot even be named here: it has no id in the new KG.)
  std::vector<float> queryE3{0.f, 0.f, 1.f, 0.f};  // <e3>'s old vector
  auto hnsw = idx.searchHnsw(queryE3, 4);
  ASSERT_EQ(hnsw.size(), 3u);
  std::set<uint64_t> liveIds{getIdNew("<e0>").getBits(),
                             getIdNew("<e1>").getBits(),
                             getIdNew("<e2>").getBits()};
  for (const auto& hit : hnsw) {
    EXPECT_TRUE(liveIds.contains(hit.entity_.getBits()));
  }

  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(basename + ".vec.clip" + suffix, ec);
  }
}

// _____________________________________________________________________________
// An IRI in QLever's reserved magic-service namespace that no service handles
// is a clear error (feature not linked / typo), not a silent federated request.
TEST(VectorSearchService, unknownMagicServiceIriRejected) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec,
                "SELECT * WHERE { SERVICE "
                "<https://qlever.cs.uni-freiburg.de/nosuchservice/> { ?s "
                "?p ?o } }"),
      HasSubstr("reserved magic-service namespace"));
  // A real federated endpoint under the same domain (with an `/api/` path) is
  // NOT treated as a magic service and parses as an ordinary SERVICE.
  EXPECT_NO_THROW(planQuery(
      qec,
      "SELECT * WHERE { SERVICE SILENT "
      "<https://qlever.cs.uni-freiburg.de/api/wikidata> { ?s ?p ?o } }"));
}

// _____________________________________________________________________________
// When the ONLY subtree binding the outer `<left>` variable also binds the
// `<result>` variable (jcs > 1), there is no completion order -- this must be a
// clear user error, not an internal planner assertion.
TEST(VectorSearchService, outerBoundLeftAndResultSameTripleRejected) {
  auto* qec = qecWithVectorIndex();
  AD_EXPECT_THROW_WITH_MESSAGE(
      planQuery(qec, std::string{PREFIX} +
                         "SELECT * WHERE { ?x <rel> ?nn . SERVICE vec: { _:c "
                         "vec:index \"clip\" ; vec:left ?x ; vec:result ?nn "
                         "; vec:k 2 . } }"),
      HasSubstr("only"));
}

// _____________________________________________________________________________
// End-to-end test of the `--service-index` plumbing: drive the registered
// build hook with a JSON spec + a `.npy`/iris input, then the load hook, then
// query -- exercising `parseSpec`, `buildFromNpy`, `IndexExtensionRegistry`,
// and `getVectorIndex` (which `qecWithVectorIndex` above bypasses).
TEST(VectorSearchService, buildAndLoadHookRoundTrip) {
  // A fresh KG (its own on-disk basename; NOT the shared cached test index).
  std::string kg =
      "<a0> <is-a> <Doc> . <a1> <is-a> <Doc> . <a2> <is-a> <Doc> .";
  Index index = ad_utility::testing::makeTestIndex("vecHookTest", kg);
  const std::string& base = index.getOnDiskBase();

  // Write a 3x4 float32 .npy + an aligned iris file for <a0>,<a1>,<a2>.
  std::string npy = base + ".input.npy";
  std::string iris = base + ".input.iris";
  std::vector<float> data{1.f, 0.f, 0.f, 0.f, 0.9f, 0.1f,
                          0.f, 0.f, 0.f, 1.f, 0.f,  0.f};
  {
    std::string dict =
        "{'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }";
    size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
    dict.append(pad, ' ');
    dict.push_back('\n');
    std::ofstream out{npy, std::ios::binary};
    out.write("\x93NUMPY", 6);
    char ver[2] = {1, 0};
    out.write(ver, 2);
    uint16_t hlen = static_cast<uint16_t>(dict.size());
    char lb[2] = {static_cast<char>(hlen & 0xff),
                  static_cast<char>((hlen >> 8) & 0xff)};
    out.write(lb, 2);
    out.write(dict.data(), dict.size());
    out.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float));
    std::ofstream irisOut{iris};
    irisOut << "<a0>\n<a1>\n<a2>\n";
  }

  auto spec = [&](const std::string& extra) {
    return nlohmann::json::parse(
        "{\"vectorSearch\":[{\"name\":\"docs\",\"npy\":\"" + npy +
        "\",\"iris\":\"" + iris + "\",\"metric\":\"cosine\"" + extra + "}]}");
  };

  ASSERT_FALSE(IndexExtensionRegistry::get().buildHooks().empty());
  ASSERT_FALSE(IndexExtensionRegistry::get().loadHooks().empty());
  // Build via every registered build hook (the vector one consumes the key).
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(index, base, spec(""));
  }
  // Load via the registered load hooks into the index's `IndexImpl`.
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(index.getImpl(), base);
  }

  auto vidx = qlever::vector::getVectorIndex(index, "docs");
  ASSERT_TRUE(vidx != nullptr);
  EXPECT_EQ(vidx->numLiveVectors(), 3u);
  auto getId = makeGetId(index);
  auto res = vidx->searchExact(std::vector<float>{1.f, 0.f, 0.f, 0.f}, 2);
  ASSERT_EQ(res.size(), 2u);
  EXPECT_EQ(res[0].entity_, getId("<a0>"));
  EXPECT_EQ(res[1].entity_, getId("<a1>"));

  // parseSpec validation surfaces through the build hook: unknown key, and the
  // i8+non-cosine rejection.
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          index, base, spec(",\"frobnicate\":1")),
      HasSubstr("Unknown key"));
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          index, base,
          nlohmann::json::parse(
              "{\"vectorSearch\":[{\"name\":\"q\",\"npy\":\"" + npy +
              "\",\"iris\":\"" + iris +
              "\",\"scalar\":\"i8\",\"metric\":\"l2sq\"}]}")),
      HasSubstr("cosine"));
  // An invalid `preload` value is rejected at build time.
  AD_EXPECT_THROW_WITH_MESSAGE(
      IndexExtensionRegistry::get().buildHooks().front()(
          index, base, spec(",\"preload\":\"bogus\"")),
      HasSubstr("preload"));

  // A build with `alignRows` + `preload` round-trips: rows are 64-byte aligned,
  // the residency preference persists, and results are identical.
  auto alignedSpec = nlohmann::json::parse(
      "{\"vectorSearch\":[{\"name\":\"docsA\",\"npy\":\"" + npy +
      "\",\"iris\":\"" + iris +
      "\",\"metric\":\"cosine\",\"alignRows\":true,\"preload\":\"advise\"}]}");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(index, base, alignedSpec);
  }
  qlever::vector::VectorIndex aligned;
  aligned.open(base,
               "docsA");  // Residency::None -> applies persisted "advise".
  EXPECT_EQ(aligned.metadata().config_.preload_, "advise");
  EXPECT_EQ(aligned.metadata().rowStrideBytes_ % 64, 0u);
  auto resA = aligned.searchExact(std::vector<float>{1.f, 0.f, 0.f, 0.f}, 2);
  ASSERT_EQ(resA.size(), 2u);
  EXPECT_EQ(resA[0].entity_, getId("<a0>"));
  EXPECT_EQ(resA[1].entity_, getId("<a1>"));

  for (auto* suffix :
       {".vec.docs.meta", ".vec.docs.keys", ".vec.docs.rowmap",
        ".vec.docs.data", ".vec.docs.iris", ".vec.docs.hnsw", ".vec.docsA.meta",
        ".vec.docsA.keys", ".vec.docsA.rowmap", ".vec.docsA.data",
        ".vec.docsA.iris", ".vec.docsA.hnsw", ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(base + suffix, ec);
  }
}
