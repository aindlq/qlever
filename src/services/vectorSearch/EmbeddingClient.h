// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H

#include <string>
#include <vector>

#include "util/CancellationHandle.h"

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

// Embed an image via the same OpenAI-compatible endpoint. `imagePayload` is
// what is sent in `input` -- a URL the endpoint fetches, or a
// `data:image/...;base64,` data URI. (Multimodal embedding servers vary in
// their exact request shape; this uses the common `input`-string convention.)
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

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_EMBEDDINGCLIENT_H
