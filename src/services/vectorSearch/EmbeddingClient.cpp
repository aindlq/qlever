// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/EmbeddingClient.h"

#include <cmath>
#include <mutex>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <utility>

#include <absl/strings/str_cat.h>

#ifdef QLEVER_WITH_LLAMACPP
#include <llama.h>
#endif
#include <boost/asio/connect.hpp>
#include <boost/asio/io_context.hpp>
#include <boost/asio/local/stream_protocol.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>

#include "util/Exception.h"
#include "util/http/HttpClient.h"
#include "util/http/HttpUtils.h"
#include "util/json.h"

namespace qlever::vector {

namespace {
namespace beast = boost::beast;
namespace http = beast::http;
namespace asio = boost::asio;

constexpr std::string_view kTarget = "/v1/embeddings";

// POST `body` to a server listening on a Unix domain socket (synchronous Beast).
// `socketPath` is the filesystem path of the socket. Used for `unix:` endpoints
// (e.g. a local vLLM/llama.cpp server) -- lower overhead than TCP and off the
// network. Returns the response body; throws on a non-200 status.
std::string postViaUnixSocket(const std::string& socketPath,
                              const std::string& body) {
  asio::io_context ioc;
  asio::local::stream_protocol::socket socket{ioc};
  socket.connect(asio::local::stream_protocol::endpoint{socketPath});

  http::request<http::string_body> req{http::verb::post,
                                       std::string{kTarget}, 11};
  req.set(http::field::host, "localhost");
  req.set(http::field::content_type, "application/json");
  req.set(http::field::accept, "application/json");
  req.body() = body;
  req.prepare_payload();
  http::write(socket, req);

  beast::flat_buffer buffer;
  http::response<http::string_body> res;
  http::read(socket, buffer, res);
  beast::error_code ec;
  socket.shutdown(asio::local::stream_protocol::socket::shutdown_both, ec);

  if (res.result() != http::status::ok) {
    AD_THROW(absl::StrCat("The embedding endpoint at unix:", socketPath,
                          " returned HTTP status ",
                          static_cast<int>(res.result_int()), ": ",
                          res.body().substr(0, 200)));
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
  urlStr += kTarget;
  ad_utility::httpUtils::Url url{urlStr};
  HttpOrHttpsResponse response = sendHttpOrHttpsRequest(
      url, std::move(handle), http::verb::post, body, "application/json",
      "application/json");
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
// In-process embedding via llama.cpp from a local GGUF model (no server). Models
// are loaded once per path and cached. Written against the llama.cpp embedding
// API; enabled only with -DQLEVER_WITH_LLAMACPP=ON.
std::vector<float> embedViaLlamaCpp(const std::string& modelPath,
                                    const std::string& text) {
  static std::once_flag backendOnce;
  std::call_once(backendOnce, [] { llama_backend_init(); });

  struct Loaded {
    llama_model* model;
    llama_context* ctx;
  };
  static std::mutex mutex;
  static std::unordered_map<std::string, Loaded> cache;
  std::lock_guard<std::mutex> lock{mutex};

  auto it = cache.find(modelPath);
  if (it == cache.end()) {
    llama_model_params mp = llama_model_default_params();
    llama_model* model = llama_model_load_from_file(modelPath.c_str(), mp);
    AD_CONTRACT_CHECK(model != nullptr, "Could not load GGUF model ", modelPath);
    llama_context_params cp = llama_context_default_params();
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    llama_context* ctx = llama_init_from_model(model, cp);
    AD_CONTRACT_CHECK(ctx != nullptr, "Could not create llama context for ",
                      modelPath);
    it = cache.emplace(modelPath, Loaded{model, ctx}).first;
  }
  llama_model* model = it->second.model;
  llama_context* ctx = it->second.ctx;
  const llama_vocab* vocab = llama_model_get_vocab(model);

  std::vector<llama_token> tokens(text.size() + 8);
  int n = llama_tokenize(vocab, text.c_str(), static_cast<int>(text.size()),
                         tokens.data(), static_cast<int>(tokens.size()),
                         /*add_special=*/true, /*parse_special=*/false);
  AD_CONTRACT_CHECK(n > 0, "Tokenization failed for the embedding input.");
  tokens.resize(n);

  llama_kv_self_clear(ctx);
  llama_batch batch = llama_batch_get_one(tokens.data(), n);
  AD_CONTRACT_CHECK(llama_decode(ctx, batch) >= 0, "llama_decode failed.");

  int dim = llama_model_n_embd(model);
  const float* emb = llama_get_embeddings_seq(ctx, 0);
  if (emb == nullptr) {
    emb = llama_get_embeddings(ctx);
  }
  AD_CONTRACT_CHECK(emb != nullptr, "llama.cpp returned no embeddings.");
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
  if (baseUrl.compare(0, prefix.size(), prefix) != 0) {
    return std::nullopt;
  }
  std::string rest = baseUrl.substr(prefix.size());
  // Drop an optional "//" empty-authority (so "unix:///p" and "unix:/p" both
  // yield "/p").
  if (rest.compare(0, 2, "//") == 0) {
    rest = rest.substr(2);
  }
  return rest;
}
}  // namespace

// Embed a single `input` value (text or image payload) via the OpenAI-compatible
// endpoint and return `data[0].embedding`.
std::vector<float> embedOneOpenAI(const std::string& baseUrl,
                                  const std::string& model,
                                  const std::string& input,
                                  ad_utility::SharedCancellationHandle handle) {
  AD_CONTRACT_CHECK(!baseUrl.empty(),
                    "This vector index has no embedding endpoint configured.");

  // `llama:/path/to/model.gguf` -> in-process llama.cpp (no HTTP).
  constexpr std::string_view llamaPrefix = "llama:";
  if (baseUrl.compare(0, llamaPrefix.size(), llamaPrefix) == 0) {
    std::string modelPath = baseUrl.substr(llamaPrefix.size());
#ifdef QLEVER_WITH_LLAMACPP
    (void)model;  // the GGUF file fully determines the model
    return embedViaLlamaCpp(modelPath, input);
#else
    AD_THROW(
        "This QLever build has no llama.cpp embedding backend. Rebuild with "
        "-DQLEVER_WITH_LLAMACPP=ON, or use an http(s)/unix embedding endpoint.");
#endif
  }

  nlohmann::json request{{"model", model},
                         {"input", nlohmann::json::array({input})}};
  std::string requestBody = request.dump();

  std::string body;
  if (auto socketPath = unixSocketPath(baseUrl); socketPath.has_value()) {
    body = postViaUnixSocket(socketPath.value(), requestBody);
  } else {
    body = postViaTcp(baseUrl, requestBody, std::move(handle));
  }

  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(body);
  } catch (const std::exception& e) {
    AD_THROW(absl::StrCat("Could not parse embedding response as JSON: ",
                          e.what()));
  }
  AD_CONTRACT_CHECK(parsed.contains("data") && parsed["data"].is_array() &&
                        !parsed["data"].empty(),
                    "Embedding response has no non-empty `data` array.");
  const auto& embedding = parsed["data"][0].at("embedding");
  AD_CONTRACT_CHECK(embedding.is_array(),
                    "Embedding response `data[0].embedding` is not an array.");
  std::vector<float> out;
  out.reserve(embedding.size());
  for (const auto& value : embedding) {
    out.push_back(value.get<float>());
  }
  return out;
}

std::vector<float> embedTextOpenAI(const std::string& baseUrl,
                                   const std::string& model,
                                   const std::string& text,
                                   ad_utility::SharedCancellationHandle handle) {
  return embedOneOpenAI(baseUrl, model, text, std::move(handle));
}

std::vector<float> embedImageOpenAI(
    const std::string& baseUrl, const std::string& model,
    const std::string& imagePayload,
    ad_utility::SharedCancellationHandle handle) {
  return embedOneOpenAI(baseUrl, model, imagePayload, std::move(handle));
}

}  // namespace qlever::vector
