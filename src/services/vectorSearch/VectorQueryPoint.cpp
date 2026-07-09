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

// Report the embedding phase of a SERVICE search: the usual timing line for a
// real round trip, or an explicit `CACHED (... ms saved)` line when the
// process-wide query-embedding cache answered (so a hit is visible in the
// log).
void logQueryEmbedding(std::string_view indexName, std::string_view modality,
                       const CachedQueryEmbedding& cached) {
  if (cached.cacheHit_) {
    AD_LOG_INFO << "vec:search[" << indexName << "]: query embedding ("
                << modality << ") CACHED (" << cached.computeMs_ << " ms saved)"
                << std::endl;
  } else {
    logVectorSearchPhase(indexName,
                         absl::StrCat("query embedding (", modality, ")"),
                         cached.computeMs_);
  }
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
    // Embed through the process-lifetime query-embedding cache: a repeat of
    // the same text (with any retrieval parameters) skips the round trip.
    const std::string& text = config.queryText_.value();
    CachedQueryEmbedding cached = embedQueryCached(
        meta.embeddingUrl_, meta.embeddingModel_, /*isImage=*/false, text,
        [&meta, &text, &handle]() {
          return embedTextOpenAI(meta.embeddingUrl_, meta.embeddingModel_, text,
                                 handle);
        });
    query = *cached.embedding_;
    logQueryEmbedding(config.indexName_, "text", cached);
  } else if (config.queryImage_.has_value()) {
    const auto& meta = vidx.metadata().config_;
    if (meta.embeddingUrl_.empty()) {
      throw std::runtime_error{absl::StrCat(
          "Vector index '", config.indexName_,
          "' has no embedding endpoint configured, so image queries cannot be "
          "used.")};
    }
    // Key the cache by the PAYLOAD actually sent to the endpoint (URL or
    // `data:` URI), so `vec:imageUrl` and raw-base64 inputs that resolve to
    // the same payload share one entry.
    std::string payload = buildImagePayload(config.queryImage_.value());
    CachedQueryEmbedding cached = embedQueryCached(
        meta.embeddingUrl_, meta.embeddingModel_, /*isImage=*/true, payload,
        [&meta, &payload, &handle]() {
          return embedImageOpenAI(meta.embeddingUrl_, meta.embeddingModel_,
                                  payload, handle);
        });
    query = *cached.embedding_;
    logQueryEmbedding(config.indexName_, "image", cached);
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
