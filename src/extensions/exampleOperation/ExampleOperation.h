// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Example custom-operation extension: an outer-bound join invoked as
//   SERVICE <http://qlever.cs.uni-freiburg.de/example/successor/> {
//     [] <.../successor/input> ?x ; <.../successor/output> ?y . }
// For each `?x` bound in the surrounding query it produces `?y = ?x + 1`. The
// join side `?x` comes from the surrounding query (see
// `IncompleteJoinOperation` in `engine/MagicServicePlanning.h`). A template for
// a real join operation (spatial/vector); copy this folder, no edits to core
// QLever.

#ifndef QLEVER_SRC_EXTENSIONS_EXAMPLEOPERATION_EXAMPLEOPERATION_H
#define QLEVER_SRC_EXTENSIONS_EXAMPLEOPERATION_EXAMPLEOPERATION_H

#include <memory>
#include <optional>

#include "engine/MagicServicePlanning.h"
#include "engine/Operation.h"
#include "engine/QueryExecutionTree.h"
#include "parser/MagicServiceQuery.h"
#include "rdfTypes/Variable.h"

namespace qlever::exampleOperation {

constexpr inline std::string_view SUCCESSOR_SERVICE_IRI =
    "<http://qlever.cs.uni-freiburg.de/example/successor/>";

// Parsed configuration of the successor SERVICE: the input (join) variable and
// the produced output variable.
struct SuccessorQuery : parsedQuery::MagicServiceQuery {
  std::optional<Variable> input_;
  std::optional<Variable> output_;

  void addParameter(const SparqlTriple& triple) override;
  void validate() const override;
  std::string_view name() const override { return "example successor join"; }
};

// The operation: an outer-bound binary join. `child_` (binding `input_`) is
// taken from the surrounding query; the result adds the `output_` column.
class SuccessorJoin : public Operation, public IncompleteJoinOperation {
 public:
  SuccessorJoin(
      QueryExecutionContext* qec, Variable input, Variable output,
      std::optional<std::shared_ptr<QueryExecutionTree>> child = std::nullopt);

  // `Operation`.
  std::vector<QueryExecutionTree*> getChildren() override;
  std::string getCacheKeyImpl() const override;
  std::string getDescriptor() const override { return "ExampleSuccessorJoin"; }
  size_t getResultWidth() const override;
  size_t getCostEstimate() override;
  uint64_t getSizeEstimateBeforeLimit() override;
  float getMultiplicity(size_t) override { return 1; }
  bool knownEmptyResult() override;
  std::unique_ptr<Operation> cloneImpl() const override;
  std::vector<ColumnIndex> resultSortedOn() const override { return {}; }
  bool isDeterministicImpl() const override { return true; }
  Result computeResult(bool requestLaziness) override;
  VariableToColumnMap computeVariableToColumnMap() const override;

  // `IncompleteJoinOperation`: `input_` is the sole join variable.
  bool isJoinConstructed() const override { return child_.has_value(); }
  bool canBindJoinVariable(const Variable& var) const override {
    return var == input_;
  }
  std::string multipleJoinVariablesError() const override {
    return "The example successor join must be the only connection to its "
           "input variable.";
  }
  std::shared_ptr<Operation> addJoinChild(
      std::shared_ptr<QueryExecutionTree> child,
      const Variable& var) const override;

 private:
  Variable input_;
  Variable output_;
  std::optional<std::shared_ptr<QueryExecutionTree>> child_;
};

}  // namespace qlever::exampleOperation

#endif  // QLEVER_SRC_EXTENSIONS_EXAMPLEOPERATION_EXAMPLEOPERATION_H
