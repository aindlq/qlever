// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "extensions/exampleOperation/ExampleOperation.h"

#include <absl/strings/str_cat.h>

#include <typeindex>

#include "engine/QueryExecutionContext.h"
#include "engine/VariableToColumnMap.h"
#include "global/Id.h"
#include "parser/MagicServiceRegistry.h"
#include "parser/SparqlTriple.h"
#include "util/Exception.h"

namespace qlever::exampleOperation {

// _____________________________________________________________________________
void SuccessorQuery::addParameter(const SparqlTriple& triple) {
  auto simple = triple.getSimple();
  std::string_view param =
      extractParameterName(simple.p_, SUCCESSOR_SERVICE_IRI);
  if (param == "input") {
    setVariable("input", simple.o_, input_);
  } else if (param == "output") {
    setVariable("output", simple.o_, output_);
  } else {
    throw parsedQuery::MagicServiceException(absl::StrCat(
        "Unknown parameter for the example successor join: <", param, ">"));
  }
}

// _____________________________________________________________________________
void SuccessorQuery::validate() const {
  if (!input_.has_value() || !output_.has_value()) {
    throw parsedQuery::MagicServiceException(
        "The example successor join requires an <input> and an <output> "
        "variable.");
  }
}

// _____________________________________________________________________________
SuccessorJoin::SuccessorJoin(
    QueryExecutionContext* qec, Variable input, Variable output,
    std::optional<std::shared_ptr<QueryExecutionTree>> child)
    : Operation{qec},
      input_{std::move(input)},
      output_{std::move(output)},
      child_{std::move(child)} {}

// _____________________________________________________________________________
std::vector<QueryExecutionTree*> SuccessorJoin::getChildren() {
  if (child_.has_value()) {
    return {child_.value().get()};
  }
  return {};
}

// _____________________________________________________________________________
size_t SuccessorJoin::getResultWidth() const {
  return child_.has_value() ? child_.value()->getResultWidth() + 1 : 2;
}

// _____________________________________________________________________________
size_t SuccessorJoin::getCostEstimate() {
  return child_.has_value() ? child_.value()->getCostEstimate() +
                                  child_.value()->getSizeEstimate()
                            : 1;
}

// _____________________________________________________________________________
uint64_t SuccessorJoin::getSizeEstimateBeforeLimit() {
  return child_.has_value() ? child_.value()->getSizeEstimate() : 1;
}

// _____________________________________________________________________________
bool SuccessorJoin::knownEmptyResult() {
  return child_.has_value() && child_.value()->knownEmptyResult();
}

// _____________________________________________________________________________
std::string SuccessorJoin::getCacheKeyImpl() const {
  if (!child_.has_value()) {
    return absl::StrCat("ExampleSuccessorJoin(incomplete) ", input_.name(), " ",
                        output_.name());
  }
  return absl::StrCat("ExampleSuccessorJoin ", input_.name(), " -> ",
                      output_.name(), " {", child_.value()->getCacheKey(), "}");
}

// _____________________________________________________________________________
std::unique_ptr<Operation> SuccessorJoin::cloneImpl() const {
  std::optional<std::shared_ptr<QueryExecutionTree>> childClone;
  if (child_.has_value()) {
    childClone = child_.value()->clone();
  }
  return std::make_unique<SuccessorJoin>(_executionContext, input_, output_,
                                         std::move(childClone));
}

// _____________________________________________________________________________
VariableToColumnMap SuccessorJoin::computeVariableToColumnMap() const {
  if (!child_.has_value()) {
    // Incomplete: expose the join variable (`input_`) and the produced
    // `output_` as possibly-undefined columns, so the planner completes us via
    // the surrounding query's `input_` binding.
    VariableToColumnMap map;
    map[input_] = makePossiblyUndefinedColumn(ColumnIndex{0});
    map[output_] = makePossiblyUndefinedColumn(ColumnIndex{1});
    return map;
  }
  VariableToColumnMap map = child_.value()->getVariableColumns();
  map[output_] =
      makeAlwaysDefinedColumn(ColumnIndex{child_.value()->getResultWidth()});
  return map;
}

// _____________________________________________________________________________
Result SuccessorJoin::computeResult([[maybe_unused]] bool requestLaziness) {
  AD_CONTRACT_CHECK(
      child_.has_value(),
      "The example successor join was never completed with its input side; "
      "bind its input variable in the surrounding query.");
  std::shared_ptr<const Result> childResult = child_.value()->getResult(false);
  const auto& childTable = childResult->idTableView();
  ColumnIndex inputCol = child_.value()->getVariableColumn(input_);

  IdTable result{childTable.numColumns() + 1,
                 getExecutionContext()->getAllocator()};
  result.resize(childTable.numRows());
  for (size_t row = 0; row < childTable.numRows(); ++row) {
    for (size_t col = 0; col < childTable.numColumns(); ++col) {
      result(row, col) = childTable(row, col);
    }
    Id x = childTable(row, inputCol);
    result(row, childTable.numColumns()) = x.getDatatype() == Datatype::Int
                                               ? Id::makeFromInt(x.getInt() + 1)
                                               : Id::makeUndefined();
  }
  return {std::move(result), resultSortedOn(),
          childResult->getCopyOfLocalVocab()};
}

// _____________________________________________________________________________
std::shared_ptr<Operation> SuccessorJoin::addJoinChild(
    std::shared_ptr<QueryExecutionTree> child, const Variable& var) const {
  AD_CONTRACT_CHECK(var == input_);
  return std::make_shared<SuccessorJoin>(_executionContext, input_, output_,
                                         std::move(child));
}

namespace {
// Self-registration: parser factory + planner handler (see
// `src/extensions/CMakeLists.txt` for the OBJECT-library link).
[[maybe_unused]] const bool registered = [] {
  parsedQuery::MagicServiceRegistry::get().addExact(
      SUCCESSOR_SERVICE_IRI,
      [](const ad_utility::triple_component::Iri&)
          -> std::shared_ptr<parsedQuery::MagicServiceQuery> {
        return std::make_shared<SuccessorQuery>();
      });
  MagicServicePlannerRegistry::get().add(
      std::type_index(typeid(SuccessorQuery)),
      [](MagicServicePlanningContext& ctx,
         parsedQuery::MagicServiceQuery& query) {
        auto& q = dynamic_cast<SuccessorQuery&>(query);
        // Build an incomplete outer-bound join; the planner completes it with
        // the surrounding query's subtree that binds the input variable.
        ctx.addLeafOperation(std::make_shared<SuccessorJoin>(
            ctx.qec(), q.input_.value(), q.output_.value()));
      });
  return true;
}();
}  // namespace

}  // namespace qlever::exampleOperation
