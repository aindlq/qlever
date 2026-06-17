// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#ifndef QLEVER_SRC_ENGINE_EMBEDDINGCLIENT_H
#define QLEVER_SRC_ENGINE_EMBEDDINGCLIENT_H

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
std::vector<float> embedTextOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& text, ad_utility::SharedCancellationHandle handle);

// Embed an image via the same OpenAI-compatible endpoint. `imagePayload` is what
// is sent in `input` -- a URL the endpoint fetches, or a `data:image/...;base64,`
// data URI. (Multimodal embedding servers vary in their exact request shape; this
// uses the common `input`-string convention.)
std::vector<float> embedImageOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& imagePayload, ad_utility::SharedCancellationHandle handle);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_ENGINE_EMBEDDINGCLIENT_H
