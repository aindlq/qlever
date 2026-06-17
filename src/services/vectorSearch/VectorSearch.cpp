// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorSearch.h"

#include <absl/strings/escaping.h>
#include <absl/strings/str_cat.h>

#include <fstream>
#include <sstream>

#include "services/vectorSearch/EmbeddingClient.h"
#include "index/IndexImpl.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "util/HashMap.h"

namespace {
// Turn an image query into the payload sent to the embedding endpoint: a URL is
// passed through (the endpoint fetches it); a file is read and base64-encoded
// into a data URI; raw base64 is wrapped into a data URI (a `data:` value is
// passed through unchanged).
std::string buildImagePayload(
    const qlever::vector::VectorSearchConfiguration::ImageQuery& image) {
  using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
  auto mimeForPath = [](const std::string& p) -> std::string {
    auto ends = [&](std::string_view s) {
      return p.size() >= s.size() &&
             p.compare(p.size() - s.size(), s.size(), s) == 0;
    };
    if (ends(".png")) return "image/png";
    if (ends(".jpg") || ends(".jpeg")) return "image/jpeg";
    if (ends(".webp")) return "image/webp";
    if (ends(".gif")) return "image/gif";
    return "application/octet-stream";
  };
  switch (image.kind_) {
    case ImageKind::Url:
      return image.value_;
    case ImageKind::File: {
      std::ifstream in{image.value_, std::ios::binary};
      AD_CONTRACT_CHECK(in.is_open(), "Could not open image file ",
                        image.value_);
      std::ostringstream ss;
      ss << in.rdbuf();
      return absl::StrCat("data:", mimeForPath(image.value_), ";base64,",
                          absl::Base64Escape(ss.str()));
    }
    case ImageKind::Base64:
      if (image.value_.compare(0, 5, "data:") == 0) {
        return image.value_;
      }
      return absl::StrCat("data:image/jpeg;base64,", image.value_);
  }
  return image.value_;
}
}  // namespace

VectorSearch::VectorSearch(QueryExecutionContext* qec,
                           qlever::vector::VectorSearchConfiguration config,
                           std::shared_ptr<QueryExecutionTree> candidates)
    : Operation{qec},
      config_{std::move(config)},
      candidates_{std::move(candidates)} {
  if (candidates_) {
    const auto& cols = candidates_->getVariableColumns();
    auto it = cols.find(config_.resultVariable_);
    AD_CONTRACT_CHECK(it != cols.end(),
                      "The nested pattern of a vector search must bind the "
                      "`<result>` variable ",
                      config_.resultVariable_.name(),
                      " (the candidate entities to search among).");
    candidateCol_ = it->second.columnIndex_;
  }
  variableColumns_[config_.resultVariable_] =
      makeAlwaysDefinedColumn(ColumnIndex{0});
  if (config_.scoreVariable_.has_value()) {
    variableColumns_[config_.scoreVariable_.value()] =
        makeAlwaysDefinedColumn(ColumnIndex{1});
  }
}

std::string VectorSearch::getDescriptor() const {
  return absl::StrCat("VectorSearch on index '", config_.indexName_,
                      "', k=", config_.k_);
}

size_t VectorSearch::getResultWidth() const {
  return config_.scoreVariable_.has_value() ? 2 : 1;
}

float VectorSearch::getMultiplicity(size_t) { return 1.0f; }

uint64_t VectorSearch::getSizeEstimateBeforeLimit() {
  // At most k results; with a candidate set, at most its size.
  if (candidates_) {
    return std::min<uint64_t>(config_.k_, candidates_->getSizeEstimate());
  }
  return config_.k_;
}

size_t VectorSearch::getCostEstimate() {
  if (candidates_) {
    // Exact scan over the candidate set + the child's own cost.
    return candidates_->getCostEstimate() + candidates_->getSizeEstimate();
  }
  return config_.k_;
}

VariableToColumnMap VectorSearch::computeVariableToColumnMap() const {
  return variableColumns_;
}

std::string VectorSearch::getCacheKeyImpl() const {
  std::string key = absl::StrCat("VECTOR_SEARCH index=", config_.indexName_,
                                 " k=", config_.k_, " algo=",
                                 static_cast<int>(config_.algorithm_),
                                 " score=", config_.scoreVariable_.has_value());
  if (config_.queryEntityIri_.has_value()) {
    absl::StrAppend(&key, " entity=", config_.queryEntityIri_.value());
  }
  if (config_.queryVector_.has_value()) {
    absl::StrAppend(&key, " vec=[");
    for (float f : config_.queryVector_.value()) {
      absl::StrAppend(&key, f, ",");
    }
    absl::StrAppend(&key, "]");
  }
  if (config_.queryText_.has_value()) {
    absl::StrAppend(&key, " text=", config_.queryText_.value());
  }
  if (config_.queryImage_.has_value()) {
    absl::StrAppend(&key, " image=",
                    static_cast<int>(config_.queryImage_.value().kind_), ":",
                    config_.queryImage_.value().value_);
  }
  if (candidates_) {
    absl::StrAppend(&key, " candidates={", candidates_->getCacheKey(), "}");
  }
  return key;
}

std::unique_ptr<Operation> VectorSearch::cloneImpl() const {
  return std::make_unique<VectorSearch>(
      getExecutionContext(), config_,
      candidates_ ? candidates_->clone() : nullptr);
}

Result VectorSearch::computeResult([[maybe_unused]] bool requestLaziness) {
  const Index& index = getExecutionContext()->getIndex();
  const qlever::vector::VectorIndex* vidx =
      qlever::vector::getVectorIndex(index, config_.indexName_);
  AD_CONTRACT_CHECK(vidx != nullptr,
                    "There is no loaded vector index named '",
                    config_.indexName_, "'.");

  IdTable idTable{getResultWidth(), getExecutionContext()->getAllocator()};

  // Resolve the query vector (explicit, embedded from text, or the vector of a
  // given entity).
  std::vector<float> query;
  if (config_.queryVector_.has_value()) {
    query = config_.queryVector_.value();
  } else if (config_.queryText_.has_value()) {
    const auto& meta = vidx->metadata().config;
    AD_CONTRACT_CHECK(!meta.embeddingUrl.empty(),
                      "Vector index '", config_.indexName_,
                      "' has no embedding endpoint configured, so `vec:queryText`"
                      " cannot be used.");
    query = qlever::vector::embedTextOpenAI(meta.embeddingUrl,
                                            meta.embeddingModel,
                                            config_.queryText_.value(),
                                            cancellationHandle_);
  } else if (config_.queryImage_.has_value()) {
    const auto& meta = vidx->metadata().config;
    AD_CONTRACT_CHECK(!meta.embeddingUrl.empty(), "Vector index '",
                      config_.indexName_,
                      "' has no embedding endpoint configured, so image queries "
                      "cannot be used.");
    query = qlever::vector::embedImageOpenAI(
        meta.embeddingUrl, meta.embeddingModel,
        buildImagePayload(config_.queryImage_.value()), cancellationHandle_);
  } else {
    TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(
        config_.queryEntityIri_.value())};
    std::optional<Id> id = tc.toValueId(index.getImpl());
    if (id.has_value()) {
      auto v = vidx->getVector(id.value());
      if (v.has_value()) {
        query.assign(v->begin(), v->end());
      }
    }
    if (query.empty()) {
      // The query entity is unknown or has no vector -> empty result.
      return {std::move(idTable), resultSortedOn(), LocalVocab{}};
    }
  }
  AD_CONTRACT_CHECK(query.size() == vidx->dimensions(),
                    "Query vector dimension (", query.size(),
                    ") does not match the index dimension (", vidx->dimensions(),
                    ").");

  using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
  AD_CONTRACT_CHECK(!(config_.algorithm_ == Algo::Hnsw && !vidx->hasHnsw()),
                    "Vector search requested the HNSW algorithm, but index '",
                    config_.indexName_, "' has no HNSW structure.");

  std::vector<qlever::vector::ScoredEntity> results;
  if (candidates_) {
    // Restricted search space: gather the candidate entities the nested pattern
    // binds and run EXACT search over just those (correct and -- for a small,
    // selective set -- faster than filtering the whole HNSW index). `<algorithm>`
    // is ignored here: over an explicit candidate set, exact is the right choice.
    std::shared_ptr<const Result> candRes = candidates_->getResult();
    const IdTable& candTable = candRes->idTable();
    std::vector<Id> candidateIds;
    candidateIds.reserve(candTable.numRows());
    ad_utility::HashSet<uint64_t> seen;
    for (size_t row = 0; row < candTable.numRows(); ++row) {
      Id id = candTable(row, candidateCol_);
      if (seen.insert(id.getBits()).second) {
        candidateIds.push_back(id);
      }
    }
    results = vidx->searchExact(query, config_.k_, candidateIds);
  } else {
    // Whole-index search.
    bool useHnsw = vidx->hasHnsw() && config_.algorithm_ != Algo::Exact;
    results = useHnsw ? vidx->searchHnsw(query, config_.k_)
                      : vidx->searchExact(query, config_.k_);
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
