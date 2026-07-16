// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// End-to-end test of the example custom-operation extension: a query using the
// successor SERVICE plans into the registered outer-bound join (completed with
// the surrounding VALUES) and computes `?y = ?x + 1`.

#include <gtest/gtest.h>

#include <memory>
#include <set>
#include <string>
#include <utility>

#include "../../util/IndexTestHelpers.h"
#include "engine/QueryExecutionTree.h"
#include "engine/QueryPlanner.h"
#include "global/Id.h"
#include "index/EncodedIriManager.h"
#include "parser/SparqlParser.h"
#include "rdfTypes/Variable.h"
#include "util/CancellationHandle.h"

namespace {

TEST(ExampleOperation, successorJoinEndToEnd) {
  auto* qec = ad_utility::testing::getQec();
  auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
  static EncodedIriManager evM;
  std::string query =
      "SELECT ?x ?y WHERE { VALUES ?x { 1 2 3 } "
      "SERVICE <http://qlever.cs.uni-freiburg.de/example/successor/> { "
      "[] <http://qlever.cs.uni-freiburg.de/example/successor/input> ?x ; "
      "<http://qlever.cs.uni-freiburg.de/example/successor/output> ?y . } }";
  ParsedQuery pq = SparqlParser::parseQuery(&evM, query);
  QueryPlanner qp{qec, handle};
  QueryExecutionTree qet = qp.createExecutionTree(pq);

  std::shared_ptr<const Result> result = qet.getResult(false);
  const auto& table = result->idTableView();
  ColumnIndex xCol = qet.getVariableColumn(Variable{"?x"});
  ColumnIndex yCol = qet.getVariableColumn(Variable{"?y"});
  ASSERT_EQ(table.numRows(), 3u);

  // Each `?x` from the surrounding VALUES gets `?y = ?x + 1`.
  std::set<std::pair<int64_t, int64_t>> got;
  for (size_t row = 0; row < table.numRows(); ++row) {
    got.emplace(table(row, xCol).getInt(), table(row, yCol).getInt());
  }
  EXPECT_EQ(got,
            (std::set<std::pair<int64_t, int64_t>>{{1, 2}, {2, 3}, {3, 4}}));
}

}  // namespace
