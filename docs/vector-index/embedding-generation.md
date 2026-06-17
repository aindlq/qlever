# Embedding generation — querying (and indexing) by text/image, not just vectors

> Companion to [`architecture.md`](./architecture.md). Captures the requirement
> that QLever be able to **generate embeddings on the fly** so a query can search
> by raw text, an existing DB literal, or an image (URL / file / base64) — not
> only by a precomputed vector. Also covers the batch path for *building* an
> index directly from text/image columns.
>
> Status: design. The core vector index (storage/search) does not depend on this;
> this layer plugs in at the query path and at index build.

## 1. Why and what

Today a similarity search needs a query *vector*. We want the user to instead
provide:

- **text** — a literal in the query (`"a bronze horse"`), or
- **an existing DB value** — bind a `?label` from the graph and embed it, or
- **an image** — by URL, local file path, or base64.

QLever turns the input into a vector with the **same model** that produced the
index, then runs the normal kNN search.

## 2. The hard invariant: query model == index model

An embedding is only comparable to vectors built by the *same* model in the
*same* space. Therefore the embedding endpoint/model is **a property of the
vector index**, recorded in its metadata (`VectorIndexMetadata`), not chosen per
query. A query may *select* which index to use (`vec:index "clip"`), but the
model used to embed the query input is dictated by that index. This prevents the
single most damaging silent error (embedding the query with the wrong model).

`VectorIndexConfig`/`VectorIndexMetadata` gain an optional `EmbeddingEndpoint`:

```jsonc
"embedding": {
  "api": "openai",                 // OpenAI-compatible /v1/embeddings
  "model": "clip-ViT-B-32",
  "endpoint": "unix:///run/vllm/clip.sock",  // or http://host:port, https://...
  "modality": "image",             // "text" | "image" | "multimodal"
  "dimensions": 512,               // must equal the index dimension
  "apiKeyEnv": "QLEVER_CLIP_API_KEY"  // name of env var holding the key (never store the key)
}
```

If an index has no `embedding` block, it is vector-only (queries must supply
`vec:queryVector` or `vec:left`).

## 3. Backend abstraction — target the OpenAI-compatible API, not "OpenAI"

The recommendation (answering "or do you have better ideas?"): standardise on the
**OpenAI-compatible embeddings protocol**, because every relevant server speaks
it — OpenAI, **vLLM**, HuggingFace **TEI**, **llama.cpp**'s `server`, **Ollama**.
One client therefore covers local and hosted, text and (multimodal) image models.

```cpp
// dependency-light interface (no engine includes)
struct EmbeddingRequestItem {
  enum class Kind { Text, ImageUrl, ImageFile, ImageBase64 } kind;
  std::string value;        // the text, URL, path, or base64 payload
};
class EmbeddingProvider {
 public:
  virtual ~EmbeddingProvider() = default;
  // Embed a batch; returns one vector per item, each of `dimensions()` floats.
  virtual std::vector<std::vector<float>> embed(
      std::span<const EmbeddingRequestItem>) = 0;
  virtual size_t dimensions() const = 0;
};
```

Two implementations:

1. **`OpenAiEmbeddingProvider` (default, production).** POSTs to
   `<endpoint>/v1/embeddings` with `{ "model": ..., "input": [...] }` and parses
   `data[i].embedding`. Built on QLever's existing **Boost.Beast** HTTP client
   (already used for federated `SERVICE`; see `src/engine/Service.cpp` /
   `src/util/http/`), which supports:
   - **TCP** (`http://`, `https://`) — hosted OpenAI, remote vLLM/TEI.
   - **Unix domain socket** (`unix:///path.sock`) — the recommended production
     setup: a local vLLM/llama.cpp/TEI server bound to a Unix socket. Lower
     overhead than TCP and not exposed on the network. Boost.Beast speaks HTTP
     over `boost::asio::local::stream_protocol`.
   API key (when needed) read from the env var named by `apiKeyEnv` — never
   stored in the index.

2. **`LlamaCppEmbeddingProvider` (optional, behind `QLEVER_WITH_LLAMACPP`).**
   Links llama.cpp and computes embeddings **in-process** (no server, no socket)
   from a local GGUF model. Nice for self-contained, offline deployments. Off by
   default to keep the dependency optional.

A small factory builds the right provider from the index's `EmbeddingEndpoint`.

### Image inputs

The OpenAI embeddings spec is text-first; multimodal/image embedding servers
(e.g. vLLM serving a CLIP model) accept images via an agreed request shape. We
normalise all image inputs to **base64** before sending (fetch URL → bytes; read
file → bytes; base64 passthrough) and send them in the field the configured model
expects (configurable; default mirrors vLLM's CLIP convention). `ImageUrl` may
alternatively be forwarded as-is when the endpoint fetches URLs itself.

## 4. SPARQL surface additions

The `vec:` SERVICE gains input predicates that trigger embedding (mutually
exclusive with `vec:queryVector` / `vec:left`):

```sparql
SERVICE vec: {
  _:cfg vec:index "clip" ;
        vec:queryText "a bronze horse statue" ;   # embed text via index "clip"
        vec:right ?artwork ; vec:k 10 ; vec:bindScore ?s .
}
```

- `vec:queryText "..."` or `vec:queryText ?dbLiteral` — embed text (constant or a
  bound DB value).
- `vec:queryImage <http://…>` / `"file:///…"` / `"data:image/png;base64,…"` —
  embed an image.

Semantics: at execution, the operation collects the (possibly per-row) inputs,
calls the index's `EmbeddingProvider::embed(...)` **once per batch** (not per
row), then proceeds exactly as if `vec:queryVector` had been supplied. Query
embeddings are cached (keyed by index+input) so repeated identical inputs in a
query/session don't re-call the model.

## 5. Indexing from text/image columns (batch)

The same provider powers building an index directly from a text/image column
instead of precomputed vectors:

```
qlever index ... --vector-index '{ "name":"clip", "dimensions":512,
  "embedding": {...}, "source": { "sparql": "SELECT ?e ?img WHERE {...}",
  "input": "?img", "inputKind": "imageUrl" } }'
```

The build pass streams `(entity, input)` rows, batches them through
`EmbeddingProvider::embed`, and feeds the resulting vectors to
`VectorIndexBuilder`. Batching + concurrency (and, for llama.cpp/vLLM, large
tensor batches) make this efficient; the same batch mechanism serves both
backends. Precomputed-vector input (Parquet/.npy) remains supported and is the
faster path when embeddings already exist.

## 6. Operational notes

- **Timeouts / retries / failure**: query-time embedding adds a network/compute
  hop; the op must honour QLever's cancellation, apply a timeout, and surface a
  clear error if the endpoint is down (don't silently return empty).
- **Dimension check**: provider `dimensions()` must equal the index dimension;
  verified at load and per call.
- **Cost**: hosted APIs cost money/latency; Unix-socket-local vLLM/llama.cpp is
  the recommended production path. Query-embedding cache mitigates repeats.
- **Security**: API keys only via env var; `unix://` sockets keep traffic off the
  network; validate/limit image file paths if exposed publicly.

## 7. Phasing (separate track from the core vector index)

- **E1** — `EmbeddingProvider` interface + `OpenAiEmbeddingProvider` (TCP first),
  `vec:queryText`; per-index `EmbeddingEndpoint` in metadata; query-time embed +
  cache. (Reuses the Beast HTTP client.)
- **E2** — Unix-socket transport; image inputs (url/file/base64).
- **E3** — batch indexing from a text/image column (`--vector-index … source`).
- **E4** — optional `LlamaCppEmbeddingProvider` behind `QLEVER_WITH_LLAMACPP`.

This track is independent of the core storage/search work and plugs into the
query path (the operation) and the build pass once those exist.
