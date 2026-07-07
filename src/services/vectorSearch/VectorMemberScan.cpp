// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorMemberScan.h"

#include <absl/strings/str_cat.h>

#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"

// ____________________________________________________________________________
VectorMemberScan::VectorMemberScan(QueryExecutionContext* qec,
                                   std::string indexName,
                                   Variable entityVariable)
    : Operation{qec},
      indexName_{std::move(indexName)},
      entityVariable_{std::move(entityVariable)} {
  variableColumns_[entityVariable_] = makeAlwaysDefinedColumn(ColumnIndex{0});
}

// ____________________________________________________________________________
std::string VectorMemberScan::getDescriptor() const {
  return absl::StrCat("VectorMemberScan on '", indexName_, "'");
}

// ____________________________________________________________________________
uint64_t VectorMemberScan::getSizeEstimateBeforeLimit() {
  // The exact member count if the index is loaded (it always is at query time);
  // a small constant while planning without a loaded index (e.g. in tests).
  auto vidx = qlever::vector::getVectorIndex(getExecutionContext()->getIndex(),
                                             indexName_);
  return vidx ? vidx->numLiveVectors() : 1;
}

// ____________________________________________________________________________
size_t VectorMemberScan::getCostEstimate() {
  // A single linear pass over the id-sorted key list: cost ~ the member count.
  return getSizeEstimateBeforeLimit();
}

// ____________________________________________________________________________
VariableToColumnMap VectorMemberScan::computeVariableToColumnMap() const {
  return variableColumns_;
}

// ____________________________________________________________________________
std::string VectorMemberScan::getCacheKeyImpl() const {
  // The produced ids depend only on the index (its member set is fixed for a
  // run); the output variable renames the column but not the data.
  return absl::StrCat("VECTOR_MEMBER_SCAN index=", indexName_);
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorMemberScan::cloneImpl() const {
  return std::make_unique<VectorMemberScan>(getExecutionContext(), indexName_,
                                            entityVariable_);
}

// ____________________________________________________________________________
Result VectorMemberScan::computeResult([[maybe_unused]] bool requestLaziness) {
  const Index& index = getExecutionContext()->getIndex();
  std::shared_ptr<const qlever::vector::VectorIndex> vidx =
      qlever::vector::getVectorIndex(index, indexName_);
  if (!vidx) {
    throw std::runtime_error{
        absl::StrCat("There is no loaded vector index named '", indexName_,
                     "'. Was the index built with `--service-index`?")};
  }
  // Emit the members as one column, already sorted in `ValueId` order (the
  // physical id-sorted rowmap order), so it merge-joins as a normal operand.
  IdTable idTable{1, getExecutionContext()->getAllocator()};
  idTable.resize(vidx->numLiveVectors());
  vidx->memberEntities(idTable.getColumn(0));
  return {std::move(idTable), resultSortedOn(), LocalVocab{}};
}
