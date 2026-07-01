// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSearch.h"

#include <absl/base/casts.h>
#include <absl/strings/str_cat.h>

#include <cmath>

#include "backports/StartsWithAndEndsWith.h"
#include "index/IndexImpl.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "services/vectorSearch/EmbeddingClient.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/HashSet.h"

namespace {
// Turn an image query into the payload sent to the embedding endpoint: a URL is
// passed through (the endpoint fetches it); raw base64 is wrapped into a data
// URI (a `data:` value is passed through unchanged).
std::string buildImagePayload(
    const qlever::vector::VectorSearchConfiguration::ImageQuery& image) {
  using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
  switch (image.kind_) {
    case ImageKind::Url:
      return image.value_;
    case ImageKind::Base64:
      if (ql::starts_with(image.value_, "data:")) {
        return image.value_;
      }
      return absl::StrCat("data:image/jpeg;base64,", image.value_);
  }
  return image.value_;
}

// Append a string to a cache key unambiguously (length-prefixed, so crafted
// values cannot collide with other parts of the key).
void appendToKey(std::string* key, std::string_view field,
                 std::string_view value) {
  absl::StrAppend(key, " ", field, "=", value.size(), ":", value);
}
}  // namespace

// ____________________________________________________________________________
VectorSearch::VectorSearch(QueryExecutionContext* qec,
                           qlever::vector::VectorSearchConfiguration config,
                           std::shared_ptr<QueryExecutionTree> candidates)
    : Operation{qec},
      config_{std::move(config)},
      candidates_{std::move(candidates)} {
  if (candidates_) {
    const auto& cols = candidates_->getVariableColumns();
    auto it = cols.find(config_.resultVariable_);
    if (it == cols.end()) {
      throw std::runtime_error{absl::StrCat(
          "The nested pattern of a vector search must bind the `<result>` "
          "variable ",
          config_.resultVariable_.name(),
          " (the candidate entities to search among).")};
    }
    candidateCol_ = it->second.columnIndex_;
    // The candidate pattern's other variables are NOT part of this operation's
    // result (each output row is a search hit, not a candidate row), so
    // exposing or silently dropping them would both be wrong. Reject them.
    if (cols.size() != 1) {
      std::string others;
      for (const auto& [var, _] : cols) {
        if (var != config_.resultVariable_) {
          absl::StrAppend(&others, others.empty() ? "" : ", ", var.name());
        }
      }
      throw std::runtime_error{absl::StrCat(
          "The nested candidate pattern of a vector search must bind only the "
          "`<result>` variable ",
          config_.resultVariable_.name(), "; it additionally binds ", others,
          ". Select the extra values in a separate pattern outside the "
          "SERVICE and join on ",
          config_.resultVariable_.name(), ".")};
    }
  }
  variableColumns_[config_.resultVariable_] =
      makeAlwaysDefinedColumn(ColumnIndex{0});
  if (config_.scoreVariable_.has_value()) {
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{1});
  }
}

// ____________________________________________________________________________
std::string VectorSearch::getDescriptor() const {
  return absl::StrCat("VectorSearch on index '", config_.indexName_,
                      "', k=", config_.k_);
}

// ____________________________________________________________________________
size_t VectorSearch::getResultWidth() const {
  return config_.scoreVariable_.has_value() ? 2 : 1;
}

// ____________________________________________________________________________
float VectorSearch::getMultiplicity(size_t) { return 1.0f; }

// ____________________________________________________________________________
uint64_t VectorSearch::getSizeEstimateBeforeLimit() {
  // At most k results; with a candidate set, at most its size.
  if (candidates_) {
    return std::min<uint64_t>(config_.k_, candidates_->getSizeEstimate());
  }
  return config_.k_;
}

// ____________________________________________________________________________
size_t VectorSearch::getCostEstimate() {
  if (candidates_) {
    // Exact scan over the candidate set + the child's own cost.
    return candidates_->getCostEstimate() + candidates_->getSizeEstimate();
  }
  // Whole-index search: a brute-force scan touches every vector, an HNSW probe
  // roughly log(N) of them. The index may not be loaded while planning (e.g.
  // in tests); fall back to a small constant then.
  auto vidx = qlever::vector::getVectorIndex(getExecutionContext()->getIndex(),
                                             config_.indexName_);
  if (!vidx) {
    return config_.k_;
  }
  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
  size_t n = vidx->numVectors();
  if (useHnsw) {
    return static_cast<size_t>(std::log2(static_cast<double>(n) + 1) + 1) *
           config_.k_;
  }
  return n;
}

// ____________________________________________________________________________
VariableToColumnMap VectorSearch::computeVariableToColumnMap() const {
  return variableColumns_;
}

// ____________________________________________________________________________
std::string VectorSearch::getCacheKeyImpl() const {
  std::string key = absl::StrCat("VECTOR_SEARCH index=", config_.indexName_,
                                 " k=", config_.k_,
                                 " algo=", static_cast<int>(config_.algorithm_),
                                 " score=", config_.scoreVariable_.has_value());
  if (config_.maxDistance_.has_value()) {
    // Bit-exact so that two nearby values never share a key.
    absl::StrAppend(
        &key, " maxDist=",
        absl::Hex(absl::bit_cast<uint32_t>(config_.maxDistance_.value())));
  }
  if (config_.queryEntityIri_.has_value()) {
    appendToKey(&key, "entity", config_.queryEntityIri_.value());
  }
  if (config_.queryVector_.has_value()) {
    absl::StrAppend(&key, " vec=[");
    for (float f : config_.queryVector_.value()) {
      // Bit-exact: `absl::StrCat(float)` would round to six significant
      // digits, making distinct query vectors collide on one key.
      absl::StrAppend(&key, absl::Hex(absl::bit_cast<uint32_t>(f)), ",");
    }
    absl::StrAppend(&key, "]");
  }
  if (config_.queryText_.has_value()) {
    appendToKey(&key, "text", config_.queryText_.value());
  }
  if (config_.queryImage_.has_value()) {
    absl::StrAppend(&key, " imageKind=",
                    static_cast<int>(config_.queryImage_.value().kind_));
    appendToKey(&key, "image", config_.queryImage_.value().value_);
  }
  if (candidates_) {
    // `candidateCol_` is essential: two children with identical cache keys can
    // still bind the result variable to different columns.
    absl::StrAppend(&key, " candidateCol=", candidateCol_, " candidates={",
                    candidates_->getCacheKey(), "}");
  }
  return key;
}

// ____________________________________________________________________________
std::unique_ptr<Operation> VectorSearch::cloneImpl() const {
  return std::make_unique<VectorSearch>(
      getExecutionContext(), config_,
      candidates_ ? candidates_->clone() : nullptr);
}

// ____________________________________________________________________________
Result VectorSearch::computeResult([[maybe_unused]] bool requestLaziness) {
  const Index& index = getExecutionContext()->getIndex();
  std::shared_ptr<const qlever::vector::VectorIndex> vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  if (!vidx) {
    throw std::runtime_error{absl::StrCat(
        "There is no loaded vector index named '", config_.indexName_,
        "'. Was the index built with `--service-index`?")};
  }

  IdTable idTable{getResultWidth(), getExecutionContext()->getAllocator()};

  // Resolve the query point: an explicit or embedded vector goes into
  // `query`; a constant entity (`vec:query <iri>`) is searched by its STORED
  // vector directly (no decode/re-encode through f32), tracked in
  // `queryEntity`.
  std::vector<float> query;
  std::optional<Id> queryEntity;
  if (config_.queryVector_.has_value()) {
    query = config_.queryVector_.value();
  } else if (config_.queryText_.has_value()) {
    const auto& meta = vidx->metadata().config_;
    if (meta.embeddingUrl_.empty()) {
      throw std::runtime_error{absl::StrCat(
          "Vector index '", config_.indexName_,
          "' has no embedding endpoint configured, so `vec:queryText` cannot "
          "be used.")};
    }
    query = qlever::vector::embedTextOpenAI(
        meta.embeddingUrl_, meta.embeddingModel_, config_.queryText_.value(),
        cancellationHandle_);
  } else if (config_.queryImage_.has_value()) {
    const auto& meta = vidx->metadata().config_;
    if (meta.embeddingUrl_.empty()) {
      throw std::runtime_error{absl::StrCat(
          "Vector index '", config_.indexName_,
          "' has no embedding endpoint configured, so image queries cannot be "
          "used.")};
    }
    query = qlever::vector::embedImageOpenAI(
        meta.embeddingUrl_, meta.embeddingModel_,
        buildImagePayload(config_.queryImage_.value()), cancellationHandle_);
  } else {
    TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(
        config_.queryEntityIri_.value())};
    std::optional<Id> id = tc.toValueId(index.getImpl());
    if (!id.has_value() || !vidx->hasVector(id.value())) {
      // The query entity is unknown or has no vector -> empty result.
      return {std::move(idTable), resultSortedOn(), LocalVocab{}};
    }
    queryEntity = id;
  }
  if (!queryEntity.has_value() && query.size() != vidx->dimensions()) {
    throw std::runtime_error{absl::StrCat(
        "The query vector has dimension ", query.size(), ", but vector index '",
        config_.indexName_, "' has dimension ", vidx->dimensions(), ".")};
  }

  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  if (config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()) {
    throw std::runtime_error{
        absl::StrCat("Vector search requested the HNSW algorithm, but index '",
                     config_.indexName_, "' has no HNSW structure.")};
  }
  auto checkInterrupt = [this]() { checkCancellation(); };

  std::vector<qlever::vector::ScoredEntity> results;
  if (candidates_) {
    // Restricted search space: gather the candidate entities the nested pattern
    // binds and run EXACT search over just those (correct and -- for a small,
    // selective set -- faster than filtering the whole HNSW index).
    // `<algorithm>` is ignored here: over an explicit candidate set, exact is
    // the right choice.
    std::shared_ptr<const Result> candRes = candidates_->getResult();
    const IdTable& candTable = candRes->idTable();
    std::vector<Id> candidateIds;
    candidateIds.reserve(candTable.numRows());
    ad_utility::HashSet<uint64_t> seen;
    for (size_t row = 0; row < candTable.numRows(); ++row) {
      if (row % 1024 == 0) {
        checkCancellation();
      }
      Id id = candTable(row, candidateCol_);
      if (seen.insert(id.getBits()).second) {
        candidateIds.push_back(id);
      }
    }
    // NOTE: an empty candidate set correctly yields an empty result here (the
    // restriction pattern matched nothing).
    ql::span<const Id> candidateSpan{candidateIds};
    results = queryEntity.has_value()
                  ? vidx->searchExactByEntity(
                        queryEntity.value(), config_.k_, candidateSpan,
                        config_.maxDistance_, checkInterrupt)
                  : vidx->searchExact(query, config_.k_, candidateSpan,
                                      config_.maxDistance_, checkInterrupt);
  } else {
    // Whole-index search (by the stored vector for the entity form -- no
    // decode/re-encode through f32).
    bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
    if (queryEntity.has_value()) {
      results =
          useHnsw
              ? vidx->searchHnswByEntity(queryEntity.value(), config_.k_,
                                         config_.maxDistance_, checkInterrupt)
              : vidx->searchExactByEntity(queryEntity.value(), config_.k_,
                                          std::nullopt, config_.maxDistance_,
                                          checkInterrupt);
    } else {
      results = useHnsw
                    ? vidx->searchHnsw(query, config_.k_, config_.maxDistance_,
                                       checkInterrupt)
                    : vidx->searchExact(query, config_.k_, std::nullopt,
                                        config_.maxDistance_, checkInterrupt);
    }
  }

  idTable.resize(results.size());
  bool withScore = config_.scoreVariable_.has_value();
  for (size_t i = 0; i < results.size(); ++i) {
    idTable(i, 0) = results[i].entity_;
    if (withScore) {
      idTable(i, 1) = Id::makeFromDouble(results[i].distance_);
    }
  }
  return {std::move(idTable), resultSortedOn(), LocalVocab{}};
}
