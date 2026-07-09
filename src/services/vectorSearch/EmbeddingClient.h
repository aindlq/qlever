// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "util/CancellationHandle.h"
#include "util/json.h"

namespace qlever::vector {

// Embed a single text via an OpenAI-compatible embeddings endpoint.
//
// POSTs `{"model": <model>, "input": [<text>]}` to `<baseUrl>/v1/embeddings`
// (the standard OpenAI shape, also spoken by vLLM, HuggingFace TEI, llama.cpp's
// server, and Ollama) and returns `data[0].embedding`. The model is dictated by
// the vector index (so query-time and index-time use the same embedding space).
//
// Throws on a missing endpoint, a non-200 response, or a malformed body.
std::vector<float> embedTextOpenAI(const std::string& baseUrl,
                                   const std::string& model,
                                   const std::string& text,
                                   ad_utility::SharedCancellationHandle handle);

// Build the vLLM multimodal embedding request body for a single image:
// `{"model": <model>, "encoding_format": "float", "messages": [{"role":
// "user", "content": [{"type": "image_url", "image_url": {"url":
// <imagePayload>}}]}]}`. Exposed separately so the exact request shape is
// unit-testable.
nlohmann::json makeImageEmbeddingRequest(const std::string& model,
                                         const std::string& imagePayload);

// Embed an image via the vLLM multimodal embedding API, POSTed to the same
// `<baseUrl>/v1/embeddings` endpoint as the text path. `imagePayload` -- an
// http(s) image URL the endpoint fetches, or a `data:image/...;base64,` data
// URI -- is sent verbatim inside a chat-style `messages` body with a single
// `image_url` content part (see `makeImageEmbeddingRequest`), which is vLLM's
// request shape for multimodal embedding models. The response is the usual
// `CreateEmbeddingResponse` body; the embedding is read from
// `data[0].embedding`.
std::vector<float> embedImageOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& imagePayload,
    ad_utility::SharedCancellationHandle handle);

// Embed many texts with ONE request (`"input": [t0, t1, ...]`) -- the
// OpenAI-compatible batch shape. Returns one embedding per input, in input
// order. Used by the index build, where a request per row would make the build
// latency-bound on the endpoint. (The in-process `llama:` backend embeds the
// batch sequentially.)
std::vector<std::vector<float>> embedBatchOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::vector<std::string>& texts,
    ad_utility::SharedCancellationHandle handle);

// The result of a cached query-embedding lookup (see `embedQueryCached`).
struct CachedQueryEmbedding {
  // The embedding; never null. Points into the shared cache entry (the
  // `shared_ptr` keeps the entry's storage alive past eviction).
  std::shared_ptr<const std::vector<float>> embedding_;
  // True iff the embedding was served from the process-wide cache, i.e. NO
  // endpoint round trip happened for this call.
  bool cacheHit_;
  // Wall time of the endpoint round trip in milliseconds: the one THIS call
  // performed on a miss, or -- on a hit -- the ORIGINAL one that produced the
  // cached value (i.e. roughly the time the hit saved).
  double computeMs_;
};

// Compute a QUERY embedding through the PROCESS-LIFETIME query-embedding
// cache: a bounded (LRU + byte budget), thread-safe cache shared by all
// queries, so repeat queries with the same embed input skip the embedding
// round trip -- typically the dominant cost of a vector-SERVICE query.
//
// The cache key is exactly (`baseUrl`, `model`, `isImage`, `input`) -- i.e.
// the embedding-endpoint identity, the modality (the same string embeds
// differently as text vs as an image URL), and the exact input -- and NOTHING
// else: retrieval parameters (`vec:k`, thresholds, ...) never enter the key,
// so the same input embeds once regardless of them. On a miss,
// `computeFunction` (which must perform the actual round trip and be
// deterministic in the key) runs WITHOUT any cache lock held; concurrent
// misses on the same key compute once and share the result (the
// `ConcurrentCache` in-progress machinery). A failed computation is not
// cached. The cache lives for the lifetime of the process (a lazily
// initialized function-local static); it is a query-time concern only and
// never touches the index or any persisted file.
CachedQueryEmbedding embedQueryCached(
    const std::string& baseUrl, const std::string& model, bool isImage,
    const std::string& input,
    const std::function<std::vector<float>()>& computeFunction);

// TESTING ONLY: drop all entries of the process-wide query-embedding cache,
// so tests start from a clean slate.
void clearQueryEmbeddingCacheForTesting();

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H
