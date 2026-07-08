// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorQueryPoint.h"

#include <absl/strings/str_cat.h>

#include <stdexcept>

#include "backports/StartsWithAndEndsWith.h"
#include "index/IndexImpl.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "services/vectorSearch/EmbeddingClient.h"
#include "services/vectorSearch/VectorIndex.h"
#include "util/Log.h"
#include "util/Timer.h"

namespace qlever::vector {

// _____________________________________________________________________________
void logVectorSearchPhase(std::string_view indexName, std::string_view phase,
                          double milliseconds,
                          std::optional<size_t> numVectors) {
  std::string count = numVectors.has_value()
                          ? absl::StrCat(" ", numVectors.value(), " vec")
                          : std::string{};
  AD_LOG_INFO << "vec:search[" << indexName << "]: " << phase << count << " "
              << milliseconds << " ms" << std::endl;
}

namespace {
// Turn an image query into the payload sent to the embedding endpoint: a URL is
// passed through (the endpoint fetches it); raw base64 is wrapped into a data
// URI (a `data:` value is passed through unchanged).
std::string buildImagePayload(
    const VectorSearchConfiguration::ImageQuery& image) {
  using ImageKind = VectorSearchConfiguration::ImageKind;
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
}  // namespace

// ____________________________________________________________________________
QueryPoint resolveQueryPoint(const VectorSearchConfiguration& config,
                             const VectorIndex& vidx, const IndexImpl& index,
                             ad_utility::SharedCancellationHandle handle) {
  // An explicit or embedded vector goes into `query`; a constant entity
  // (`vec:query <iri>`) is searched by its STORED vector directly (no
  // decode/re-encode through f32) and returned as its `Id`.
  std::vector<float> query;
  if (config.queryVector_.has_value()) {
    query = config.queryVector_.value();
  } else if (config.queryText_.has_value()) {
    const auto& meta = vidx.metadata().config_;
    if (meta.embeddingUrl_.empty()) {
      throw std::runtime_error{absl::StrCat(
          "Vector index '", config.indexName_,
          "' has no embedding endpoint configured, so `vec:queryText` cannot "
          "be used.")};
    }
    ad_utility::Timer embedTimer{ad_utility::Timer::Started};
    query = embedTextOpenAI(meta.embeddingUrl_, meta.embeddingModel_,
                            config.queryText_.value(), std::move(handle));
    logVectorSearchPhase(config.indexName_, "query embedding (text)",
                         embedTimer.value().count() / 1000.0);
  } else if (config.queryImage_.has_value()) {
    const auto& meta = vidx.metadata().config_;
    if (meta.embeddingUrl_.empty()) {
      throw std::runtime_error{absl::StrCat(
          "Vector index '", config.indexName_,
          "' has no embedding endpoint configured, so image queries cannot be "
          "used.")};
    }
    ad_utility::Timer embedTimer{ad_utility::Timer::Started};
    query = embedImageOpenAI(meta.embeddingUrl_, meta.embeddingModel_,
                             buildImagePayload(config.queryImage_.value()),
                             std::move(handle));
    logVectorSearchPhase(config.indexName_, "query embedding (image)",
                         embedTimer.value().count() / 1000.0);
  } else {
    TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(
        config.queryEntityIri_.value())};
    std::optional<Id> id = tc.toValueId(index);
    if (!id.has_value() || !vidx.hasVector(id.value())) {
      // Unknown or vectorless query entity -> empty result.
      return std::monostate{};
    }
    return id.value();
  }
  if (query.size() != vidx.dimensions()) {
    throw std::runtime_error{absl::StrCat(
        "The query vector has dimension ", query.size(), ", but vector index '",
        config.indexName_, "' has dimension ", vidx.dimensions(), ".")};
  }
  return query;
}

}  // namespace qlever::vector
