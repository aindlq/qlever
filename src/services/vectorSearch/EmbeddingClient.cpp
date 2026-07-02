// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/EmbeddingClient.h"

#include <absl/strings/str_cat.h>

#include <chrono>
#include <cmath>
#include <mutex>
#include <optional>
#include <string_view>
#include <utility>

#ifdef QLEVER_WITH_LLAMACPP
#include <llama.h>
#endif

#include "backports/StartsWithAndEndsWith.h"
#include "util/Exception.h"
#include "util/HashMap.h"
#include "util/http/HttpClient.h"
#include "util/http/HttpUtils.h"
#include "util/http/beast.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;

constexpr std::string_view EMBEDDINGS_PATH = "/v1/embeddings";

// Upper bound on each blocking phase (connect/write/read) of a unix-socket
// embedding request. The TCP path is interruptible through the cancellation
// handle instead; for the synchronous unix-socket exchange this deadline is
// what keeps a stalled local embedding server from blocking a query worker
// thread forever.
constexpr std::chrono::seconds UNIX_SOCKET_TIMEOUT{60};

// POST `body` to a server listening on a Unix domain socket. `socketPath` is
// the filesystem path of the socket. Used for `unix:` endpoints (e.g. a local
// vLLM/llama.cpp server) -- lower overhead than TCP and off the network.
// Returns the response body; throws on connection errors, timeout, or a
// non-200 status.
std::string postViaUnixSocket(const std::string& socketPath,
                              const std::string& body) {
  asio::io_context ioc;
  beast::basic_stream<asio::local::stream_protocol> stream{ioc};
  http::response<http::string_body> res;
  try {
    stream.expires_after(UNIX_SOCKET_TIMEOUT);
    stream.socket().connect(asio::local::stream_protocol::endpoint{socketPath});

    http::request<http::string_body> req{http::verb::post,
                                         std::string{EMBEDDINGS_PATH}, 11};
    req.set(http::field::host, "localhost");
    req.set(http::field::content_type, "application/json");
    req.set(http::field::accept, "application/json");
    req.body() = body;
    req.prepare_payload();
    stream.expires_after(UNIX_SOCKET_TIMEOUT);
    http::write(stream, req);

    beast::flat_buffer buffer;
    stream.expires_after(UNIX_SOCKET_TIMEOUT);
    http::read(stream, buffer, res);
    beast::error_code ec;
    stream.socket().shutdown(
        asio::local::stream_protocol::socket::shutdown_both, ec);
  } catch (const boost::system::system_error& e) {
    AD_THROW(absl::StrCat("Could not reach the embedding endpoint at unix:",
                          socketPath, ": ", e.what()));
  }

  if (res.result() != http::status::ok) {
    AD_THROW(absl::StrCat(
        "The embedding endpoint at unix:", socketPath, " returned HTTP status ",
        static_cast<int>(res.result_int()), ": ", res.body().substr(0, 200)));
  }
  return std::move(res).body();
}

// POST `body` to a TCP/HTTP(S) endpoint via QLever's shared HTTP client.
std::string postViaTcp(const std::string& baseUrl, const std::string& body,
                       ad_utility::SharedCancellationHandle handle) {
  std::string urlStr = baseUrl;
  while (!urlStr.empty() && urlStr.back() == '/') {
    urlStr.pop_back();
  }
  urlStr += EMBEDDINGS_PATH;
  ad_utility::httpUtils::Url url{urlStr};
  HttpOrHttpsResponse response =
      sendHttpOrHttpsRequest(url, std::move(handle), http::verb::post, body,
                             "application/json", "application/json");
  if (response.status_ != http::status::ok) {
    AD_THROW(absl::StrCat("The embedding endpoint at ", urlStr,
                          " returned HTTP status ",
                          static_cast<int>(response.status_), ": ",
                          std::move(response).readResponseHead(200)));
  }
  std::string out;
  for (const auto& bytes : response.body_) {
    out.append(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  return out;
}

#ifdef QLEVER_WITH_LLAMACPP
// In-process embedding via llama.cpp from a local GGUF model (no server).
// Models are loaded once per path and cached. Written against the llama.cpp
// embedding API; enabled only with -DQLEVER_WITH_LLAMACPP=ON.
std::vector<float> embedViaLlamaCpp(const std::string& modelPath,
                                    const std::string& text) {
  static std::once_flag backendOnce;
  std::call_once(backendOnce, [] { llama_backend_init(); });

  struct Loaded {
    llama_model* model;
    llama_context* ctx;
  };
  static std::mutex mutex;
  static ad_utility::HashMap<std::string, Loaded> cache;
  std::lock_guard<std::mutex> lock{mutex};

  auto it = cache.find(modelPath);
  if (it == cache.end()) {
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mp);
    if (model == nullptr) {
      AD_THROW(absl::StrCat("Could not load the GGUF model ", modelPath));
    }
    llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (ctx == nullptr) {
      AD_THROW(
          absl::StrCat("Could not create a llama context for ", modelPath));
    }
    it = cache.emplace(modelPath, Loaded{model, ctx}).first;
  }
  llama_model* model = it->second.model;
  llama_context* ctx = it->second.ctx;
  const llama_vocab* vocab = llama_model_get_vocab(model);

  std::vector<llama_token> tokens(text.size() + 8);
  int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()),
                         tokens.data(), static_cast<int>(tokens.size()),
                         /*add_special=*/true, /*parse_special=*/false);
  if (n <= 0) {
    AD_THROW("Tokenization failed for the embedding input.");
  }
  tokens.resize(n);

  llama_kv_self_clear(ctx);
  llama_batch batch = llama_batch_get_one(tokens.data(), n);
  if (llama_decode(ctx, batch) < 0) {
    AD_THROW("llama_decode failed for the embedding input.");
  }

  int dim = llama_model_n_embd(model);
  const float* emb = llama_get_embeddings_seq(ctx, 0);
  if (emb == nullptr) {
    emb = llama_get_embeddings(ctx);
  }
  if (emb == nullptr) {
    AD_THROW("llama.cpp returned no embeddings.");
  }
  std::vector<float> out(emb, emb + dim);
  // L2-normalize (cosine-ready).
  double norm = 0.0;
  for (float v : out) norm += static_cast<double>(v) * v;
  norm = std::sqrt(norm);
  if (norm > 0) {
    for (float& v : out) v = static_cast<float>(v / norm);
  }
  return out;
}
#endif  // QLEVER_WITH_LLAMACPP

// Extract the Unix socket path from a `unix:`/`unix://` URL, or nullopt for a
// regular http(s) URL.
std::optional<std::string> unixSocketPath(const std::string& baseUrl) {
  constexpr std::string_view prefix = "unix:";
  if (!ql::starts_with(baseUrl, prefix)) {
    return std::nullopt;
  }
  std::string rest = baseUrl.substr(prefix.size());
  // Drop an optional "//" empty-authority (so "unix:///p" and "unix:/p" both
  // yield "/p").
  if (ql::starts_with(rest, "//")) {
    rest = rest.substr(2);
  }
  return rest;
}

// Embed `inputs` (texts or image payloads) via the OpenAI-compatible endpoint
// with a single request; returns one embedding per input, in input order. The
// response's per-item `index` field is honoured where present.
std::vector<std::vector<float>> embedManyOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::vector<std::string>& inputs,
    ad_utility::SharedCancellationHandle handle) {
  if (baseUrl.empty()) {
    AD_THROW("This vector index has no embedding endpoint configured.");
  }
  if (inputs.empty()) {
    return {};
  }

  // `llama:/path/to/model.gguf` -> in-process llama.cpp (no HTTP).
  constexpr std::string_view llamaPrefix = "llama:";
  if (ql::starts_with(baseUrl, llamaPrefix)) {
    std::string modelPath = baseUrl.substr(llamaPrefix.size());
#ifdef QLEVER_WITH_LLAMACPP
    (void)model;  // the GGUF file fully determines the model
    std::vector<std::vector<float>> out;
    out.reserve(inputs.size());
    for (const auto& input : inputs) {
      handle->throwIfCancelled();
      out.push_back(embedViaLlamaCpp(modelPath, input));
    }
    return out;
#else
    AD_THROW(
        "This QLever build has no llama.cpp embedding backend. Rebuild with "
        "-DQLEVER_WITH_LLAMACPP=ON, or use an http(s)/unix embedding "
        "endpoint.");
#endif
  }

  nlohmann::json request{{"model", model}, {"input", inputs}};
  std::string requestBody = request.dump();

  std::string body;
  if (auto socketPath = unixSocketPath(baseUrl); socketPath.has_value()) {
    handle->throwIfCancelled();
    body = postViaUnixSocket(socketPath.value(), requestBody);
    handle->throwIfCancelled();
  } else {
    body = postViaTcp(baseUrl, requestBody, std::move(handle));
  }

  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(body);
  } catch (const std::exception& e) {
    AD_THROW(
        absl::StrCat("Could not parse embedding response as JSON: ", e.what()));
  }
  if (!parsed.contains("data") || !parsed["data"].is_array() ||
      parsed["data"].size() != inputs.size()) {
    AD_THROW(absl::StrCat(
        "The embedding response's `data` array must contain one entry per "
        "input (",
        inputs.size(), " expected)."));
  }
  std::vector<std::vector<float>> out(inputs.size());
  size_t position = 0;
  for (const auto& item : parsed["data"]) {
    size_t index = item.value("index", position);
    if (index >= out.size() || !out[index].empty()) {
      AD_THROW(
          "The embedding response has invalid or duplicate `index` "
          "fields.");
    }
    const auto& embedding = item.at("embedding");
    if (!embedding.is_array() || embedding.empty()) {
      AD_THROW("An `embedding` in the response is not a non-empty array.");
    }
    out[index].reserve(embedding.size());
    for (const auto& value : embedding) {
      float f = value.get<float>();
      // A non-finite value would poison every distance comparison it enters.
      if (!std::isfinite(f)) {
        AD_THROW("The embedding response contains a non-finite value.");
      }
      out[index].push_back(f);
    }
    ++position;
  }
  return out;
}
}  // namespace

// ____________________________________________________________________________
std::vector<float> embedTextOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& text, ad_utility::SharedCancellationHandle handle) {
  return std::move(
      embedManyOpenAI(baseUrl, model, {text}, std::move(handle)).front());
}

// ____________________________________________________________________________
std::vector<float> embedImageOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& imagePayload,
    ad_utility::SharedCancellationHandle handle) {
  return std::move(
      embedManyOpenAI(baseUrl, model, {imagePayload}, std::move(handle))
          .front());
}

// ____________________________________________________________________________
std::vector<std::vector<float>> embedBatchOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::vector<std::string>& texts,
    ad_utility::SharedCancellationHandle handle) {
  return embedManyOpenAI(baseUrl, model, texts, std::move(handle));
}

}  // namespace qlever::vector
