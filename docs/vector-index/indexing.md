# Indexing vectors with QLever (add + build a vector index)

This walks through turning a knowledge graph plus a set of embeddings into a
QLever index you can query with `vec:distance`. See
[`index-payload-design.md`](index-payload-design.md) for the design; a vector is
index payload you search, not a triple you store, so the embeddings go in as a
**binary `.npy` bundle**, not as RDF.

## 0. Build the Docker image

```bash
# from the repo root (targets any CPU; uses the Xeon's AVX-512 at run time)
docker build -f Dockerfile.vector -t qlever-vector .
```

The image contains `qlever-index` (the indexer), `qlever-server`, and the
`qlever` CLI. On an Intel Xeon Gold 5412U (Sapphire Rapids) the NumKong distance
kernels dispatch at run time to the AVX-512 FP16/BF16/VNNI paths — no `-march`
needed. (Sanity check inside the builder stage:
`nm libnumkongDispatch.a | grep nk_dot_f32_` should list `haswell`/`skylake`
variants, not just the serial one.)

Run it with your data directory mounted at `/data`:

```bash
docker run -it --rm -v "$PWD:/data" qlever-vector bash
# you are now in /data inside the container; qlever-index / qlever-server are on PATH
```

## 1. Prepare the knowledge graph

Any RDF file QLever accepts (Turtle / N-Triples). The entities you attach
embeddings to must appear in it. Example `kg.ttl`:

```turtle
@prefix : <http://example.org/> .
:doc1 a :Document ; :title "A red bicycle" .
:doc2 a :Document ; :title "A blue car" .
:doc3 a :Document ; :title "A mountain landscape" .
```

## 2. Prepare the vectors: the `.npy` bundle

Per index you provide two row-aligned files:

- `<name>.npy` — an `N × D` **C-order** matrix (row *i* = the vector for
  entity *i*), either **float32** or **bfloat16** (the `ml_dtypes` dtype; see
  the bf16 note below). Real embedding dims (384, 512, 768, 1024, 1536, …) are
  all ×64, so the store is SIMD-aligned for free.
- `<name>.iris` — `N` lines, **one entity IRI per line** (bare, e.g.
  `http://example.org/doc1`; the `<...>` iriref form is also accepted), aligned
  with the matrix rows. Each IRI must exist in the KG (it is resolved against
  the vocabulary).

Produce them from any embedding pipeline, e.g. in Python:

```python
import numpy as np

iris    = ["http://example.org/doc1",
           "http://example.org/doc2",
           "http://example.org/doc3"]
vectors = np.asarray(model.encode(titles), dtype=np.float32)   # shape (N, D)

np.save("emb.npy", vectors)                                    # -> emb.npy
with open("emb.iris", "w") as f:
    f.write("\n".join(iris) + "\n")                            # -> emb.iris
```

> Rows whose IRI is not in the KG are skipped with a warning; entities in the KG
> with no row simply have no vector (they return UNDEF from `vec:distance`).

## 3. Build the index

`qlever-index` takes the usual options plus `--service-index`, a JSON object
keyed by service; the vector-search service reads its `"vectorSearch"` array:

```bash
qlever-index \
  -i myindex \
  -f kg.ttl -F ttl \
  --service-index '{
    "vectorSearch": [
      { "name": "emb", "npy": "emb.npy", "iris": "emb.iris",
        "metric": "cosine", "hnsw": true }
    ]
  }'
```

This builds the normal QLever index, then the vector index `emb` from the bundle
(resolving each IRI to an entity id, storing the vectors, and — because
`"hnsw": true` — building the HNSW graph). You can list several indices in the
array (e.g. one for image embeddings, one for text).

### `vectorSearch` spec keys

| key       | meaning                                                        |
|-----------|----------------------------------------------------------------|
| `name`    | index name; the index IRI is `<…/vectorSearch/index/name>` (used in `vec:distance`/`vec:embed` and carrying the metadata triples) |
| `npy`     | path to the `N × D` `.npy` matrix (float32 or `ml_dtypes` bfloat16) |
| `iris`    | path to the row-aligned IRI list                               |
| `metric`  | `cosine` \| `dot` \| `l2` (distance; smaller = closer)         |
| `hnsw`    | `true` builds the ANN graph; `false` = exact/flat store only   |
| `scalar`  | *(optional)* storage precision `f32` (default) \| `f16` \| `bf16` \| `i8`|
| `embeddingUrl`   | *(optional)* embedding endpoint bound to this index (see below) |
| `embeddingModel` | *(optional)* model name sent to that endpoint; also names the index's embedding space for the typed-query-vector comparability check (see "Typed query vectors") |

- bf16 — for bf16 embeddings save a **native 2-byte bf16 `.npy`** via
  [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes):
  `import ml_dtypes; np.save("emb.npy", vectors.astype(ml_dtypes.bfloat16))`,
  and set `"scalar": "bf16"` — a straight 2-byte store with no fp32 round-trip
  in the file, at half the file size. (An fp32 `.npy` plus `"scalar": "bf16"`
  also works; the values are truncated to bf16 at store time.)

### Embedding endpoint (`embeddingUrl` + `embeddingModel`)

To embed queries at *query time* (the `vec:embed` function, and the SERVICE's
`vec:queryText`/`vec:imageUrl`), bind an OpenAI-compatible embeddings endpoint
to the index in the build spec:

```json
{ "name": "emb", "npy": "emb.npy", "iris": "emb.iris", "metric": "cosine",
  "embeddingUrl": "http://embedder:8080", "embeddingModel": "clip" }
```

- `embeddingUrl` accepts `http://…` / `https://…` base URLs (the client POSTs
  to `<url>/v1/embeddings` — the shape spoken by vLLM, HuggingFace TEI,
  llama.cpp's server, and Ollama) as well as `unix:/path/to.sock` for a local
  Unix-domain socket.
- `embeddingModel` is the model name sent with each request. Because both are
  properties of the *index*, a query is always embedded with the model that
  produced the stored vectors, and no endpoint URL ever appears in a query.
- Text is embedded with the plain OpenAI shape (`{"model": …, "input": […]}`).
  IMAGE embedding (`vec:embed` on an IRI, and the SERVICE's `vec:imageUrl`)
  instead uses **vLLM's multimodal embedding format** on the same
  `/v1/embeddings` path: a chat-style `messages` body with an `image_url`
  content part (`{"model": …, "encoding_format": "float", "messages":
  [{"role": "user", "content": [{"type": "image_url", "image_url": {"url":
  <image URL or data URI>}}]}]}`). So for image queries, the endpoint must be
  a vLLM-style multimodal embedding server.
- Without `embeddingUrl`, the index is fully usable with explicit query
  vectors and entity↔entity distances; only `vec:embed` (and the SERVICE's
  text/image query points) error with "has no embeddingUrl configured".
- Both keys are **optional at build time for the `npy`/`parquet` inputs**
  (only the `texts` input needs `embeddingUrl`, to embed at build time) and
  are persisted in the index's `.meta` file when given. They can also be
  **set or changed at server start — no rebuild needed** — via the environment
  variable `QLEVER_VECTOR_SEARCH_ENDPOINTS`: a JSON object keyed by index
  name, each value with optional `embeddingUrl`/`embeddingModel` fields:

  ```bash
  QLEVER_VECTOR_SEARCH_ENDPOINTS='{"images": {"embeddingUrl": "unix:/siglip2.private", "embeddingModel": "siglip"},
                                   "metadata": {"embeddingUrl": "unix:/qwen3.private"}}' \
    qlever-server -i myindex -p 7001
  ```

  Only the fields present are overridden (a URL-only override keeps the
  persisted model, and vice versa). The override is **in-memory only** — the
  on-disk `.meta` is never rewritten — so it is reapplied on every server
  start, and starting without the variable falls back to the persisted
  endpoint. A malformed value is logged as a warning and ignored; it never
  prevents the server from starting.

## 4. Serve and query

```bash
qlever-server -i myindex -p 7001 &
```

Rank entities by distance to a query vector — plain SPARQL, exact, no SERVICE.
The index is addressed by its **IRI** `<…/vectorSearch/index/emb>` — the same
resource that carries the queryable metadata triples:

```sparql
PREFIX vec:  <https://qlever.cs.uni-freiburg.de/vectorSearch/>
PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>
SELECT ?doc ?dist WHERE {
  ?doc a <http://example.org/Document> .
  BIND(vec:distance(vidx:emb, ?doc, "0.12,-0.03, … ,0.44") AS ?dist)  # a query vector, D comma-separated floats
  FILTER(BOUND(?dist))                                                # drop entities with no vector
}
ORDER BY ?dist
LIMIT 10
```

`vec:distance(vidx:emb, S1, S2)` takes two vector *sources*; each is an
**entity** (its stored vector is looked up) or a **comma-separated float-list
string**. A source that is neither — including an entity without a vector — 
makes the row `UNDEF`, so `FILTER(BOUND(…))` prunes it. That one function
covers all the query patterns:

```sparql
# entity <-> entity: "how similar are <A> and <B>?" (both stored vectors)
BIND(vec:distance(vidx:emb, <http://example.org/doc1>,
                            <http://example.org/doc3>) AS ?dist)

# text search: embed the query via the index's own endpoint (`embeddingUrl`),
# then rank by distance -- vec:embed returns a TYPED float-list literal
BIND(vec:distance(vidx:emb, ?doc,
                  vec:embed(vidx:emb, "a red bicycle")) AS ?dist)

# image search: an IRI input to vec:embed is an image URL for the endpoint
BIND(vec:distance(vidx:emb, ?doc,
                  vec:embed(vidx:emb, <https://example.org/cat.jpg>)) AS ?dist)
```

- A constant source (an inline float list, a constant entity, a `vec:embed`
  of a constant) is resolved **once** per query, not per row; a constant
  `vec:embed` input hits the endpoint exactly once (and per-row inputs are
  memoized per distinct value).
- The metric is a property of the index, so distances are uniformly
  smaller-is-closer → always `ORDER BY ?dist ASC`.
- For a huge, *unfiltered* whole-index top-k, use the HNSW `SERVICE` instead of
  the brute-force scan above (see the design doc). A selective filter on the
  entity (like `?doc a :Document`) keeps the exact brute-force path fast.

### Typed query vectors & cross-index search (`vec:vector`)

`vec:embed` returns its vector as a **typed literal**

```
"f0,f1,…"^^<https://qlever.cs.uni-freiburg.de/vectorSearch/vec/MODEL/PRECISION>
```

(e.g. `…/vec/clip/f32`), where `MODEL` is the index's `embeddingModel` (empty
for a vector-only index) and `PRECISION` its storage `scalar` — the datatype
carries the **embedding space** the vector lives in.
`vec:vector(<…/index/NAME>, entity)` returns the entity's **stored** vector
from that index, decoded to f32, in the same typed form (UNDEF for an entity
that is not in the index).

`vec:distance` **validates** a typed query vector against the index it is
called on — the vector is used only if

- its `PRECISION` equals the index's storage scalar,
- its `MODEL` equals the index's `embeddingModel` (skipped when **either**
  side declares no model, so model-less indices keep working), and
- its float count equals the index's dimension.

A mismatching per-row value makes that row **UNDEF** — never a silently wrong
number; a mismatching *constant* throws a clear error (it is certainly a query
mistake). A plain **untyped** float string is only dimension-checked, so
inline query vectors like `"0.1,0.2,…"` keep working.

This makes **cross-index** distance safe — compare entities of one index
against a vector fetched from another, iff the two indices share an embedding
space (same model + precision + dimension):

```sparql
# artworks similar to a photo: ?p's stored vector comes from vidx:photo and
# carries that index's space; vec:distance checks it against vidx:artwork
BIND(vec:distance(vidx:artwork, ?a, vec:vector(vidx:photo, ?p)) AS ?d)
```

Note the simplest layout needs no cross-index step at all: an index is
entity-keyed, so entities of different kinds embedded with the **same model
can live in one index**, where entity↔entity distance is direct
(`vec:distance(vidx:all, ?a, ?p)`). FUTURE option (documented, deliberately
not implemented): a `matryoshka: true` index flag that lets a *longer*
Matryoshka-trained query vector be auto-truncated to a shorter index's
dimension instead of being rejected by the dimension check.

## 5. Re-indexing

Re-run the same `qlever-index` command whenever the KG or the vectors change.
The `.npy` bundle is the source of truth for the vectors; the KG is the source
for the entities they attach to.
