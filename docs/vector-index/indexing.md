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
| `scalar`  | *(optional)* storage precision `f32` (default) \| `f16` \| `bf16` \| `i8` \| `binary` (1-bit sign-packed, see below) |
| `rerank`  | *(optional)* second-layer **rerank precision** `bf16` \| `f16` \| `f32`; unset = single-layer. Builds a TWO-LAYER quantize+rerank store (see below) |
| `csls`    | *(optional, cosine-only)* `true` precomputes the per-document **CSLS hubness `r(d)`** sidecar, enabling the query-time `vec:cslsThreshold` cut (see below) |
| `cslsNeighbors` | *(optional, default 10)* neighbour count `k` of the CSLS r-terms — both the build-time `r(d)` and the query-time `r(q)` average the top-`k` cosine similarities. Persisted in the `.meta` |
| `cslsR`   | *(optional)* path to a **precomputed `r(d)`** as a float32 `.npy` of shape `(N,)` (or `(N, 1)`), row-aligned with the `npy` input — the "GPU path"; skips the build-time self-kNN |
| `embeddingUrl`   | *(optional)* embedding endpoint bound to this index (see below) |
| `embeddingModel` | *(optional)* model name sent to that endpoint; also names the index's embedding space for the typed-query-vector comparability check (see "Typed query vectors") |

- bf16 — for bf16 embeddings save a **native 2-byte bf16 `.npy`** via
  [`ml_dtypes`](https://github.com/jax-ml/ml_dtypes):
  `import ml_dtypes; np.save("emb.npy", vectors.astype(ml_dtypes.bfloat16))`,
  and set `"scalar": "bf16"` — a straight 2-byte store with no fp32 round-trip
  in the file, at half the file size. (An fp32 `.npy` plus `"scalar": "bf16"`
  also works; the values are truncated to bf16 at store time.)

### Two-layer quantize + rerank (`rerank`)

```json
{ "name": "emb", "npy": "emb.npy", "iris": "emb.iris",
  "metric": "cosine", "scalar": "i8", "rerank": "bf16", "hnsw": true }
```

With `rerank` set, the builder writes — from the **same** input matrix — two
flat stores with identical row order:

- `‹base›.vec.‹name›.data` — the coarse **scan** layer in `scalar` (here
  `i8`, the existing quantization path, unchanged). This is what brute-force
  candidate scans and the HNSW graph read.
- `‹base›.vec.‹name›.rerank.data` — the fine **rerank** layer at the `rerank`
  precision (`bf16`/`f16`/`f32`; never `i8`/`binary` — it is the
  high-precision layer).

Storage is the sum of the two layers (`i8` + `bf16` = 3 bytes per dimension —
still 25 % less than a single `f32` store). Both layers carry the same metric;
`i8` keeps its cosine-only restriction. The rerank precision is recorded in
the `.meta` (`rerankScalar`); a `.meta` without the field is a normal
single-layer index, so **existing indices load unchanged**.

#### The `binary` scan layer (1 bit per component)

`"scalar": "binary", "rerank": "bf16"` is the extreme rung of the same
two-layer idea: the scan layer keeps only the **sign bit** of each component
(bit set iff the component `> 0`, packed 8 per byte — a row is
`⌈dim / 8⌉` bytes, e.g. **dim 768 = 96 bytes per vector**, 32× smaller than
`f32`, 8× smaller than `i8`), and the coarse pass ranks candidates by the
**Hamming distance** (the number of differing sign bits — an *angular proxy*,
which is why `binary`, like `i8`, is **cosine-only**). The HNSW graph (if
requested) is likewise built over the packed sign bits with the Hamming
metric. Because 1-bit ranking is much coarser than `i8`, the default
`vec:rerankK` widens from `max(10·k, 100)` to `max(50·k, 500)`.

A `binary` index **wants a rerank layer**: without one, *every* distance the
index can serve — searches, `vec:distance`, `vec:bindScore` — is the integer
Hamming proxy, never an exact cosine (allowed, and the build logs a warning).
And note the coarse-score caveat: on a binary index `vec:bindCoarseScore` is
the raw Hamming distance (an integer in `[0, dim]`), which lives on a
*different scale* than the fine cosine score — `ABS(?d - ?dc)` is **not** the
quantization error there (unlike on an `i8` scan layer).

What each layer serves at query time:

- **`vec:distance` (and every exact primitive) reads the fine layer** — it
  stays the exact baseline, as if you had built a plain `bf16` index.
- The **`SERVICE` top-k runs coarse-scan-then-rerank**: top-`rerankK`
  candidates off the cheap `i8` bytes, then their distances are recomputed
  exactly on the `bf16` layer and the top `k` by fine distance are returned
  (see `usage-and-comparison.md`, use case 7).

### CSLS hub suppression (`csls`, `cslsNeighbors`, `cslsR`)

```json
{ "name": "emb", "npy": "emb.npy", "iris": "emb.iris",
  "metric": "cosine", "hnsw": true, "csls": true, "cslsNeighbors": 10 }
```

CSLS (Cross-domain Similarity Local Scaling) turns the SERVICE's top-k into a
**query-adaptive cut** that also suppresses *hub* documents (vectors that are
near everything). At build time, `"csls": true` precomputes each document's
**hubness**

> `r(d)` = the mean **cosine similarity** of `d`'s stored vector to its
> `cslsNeighbors` nearest corpus neighbours, **self-excluded**

and stores it as the row-aligned f32 sidecar `‹base›.vec.‹name›.csls` (it is
row-keyed like `.data`, so a cheap remap keeps it valid). At query time,
`vec:cslsThreshold τ` then keeps a candidate `d` iff

> `CSLS(q, d) = 2·cos_sim(q, d) − r(q) − r(d) ≥ τ`

(see `usage-and-comparison.md` for the query surface). How `r(d)` is computed:

- **`hnsw: true`** (recommended): a *self-kNN* against the just-built graph.
  Because `r(d)` is computed **once** and then feeds **every** query, a `csls`
  index is tuned for recall, not build speed: unless you override them, the
  graph defaults rise to **`hnswConnectivity` (M) = 32** and
  **`hnswExpansionAdd` (efConstruction) = 256** (vs. the usual 16 / 128), and
  the self-kNN searches with a **high expansion** `max(1024, 100·cslsNeighbors,
  4·fetch)` — far above the query-time `hnswExpansionSearch`, since efSearch
  can't exceed the graph's recall ceiling. On a two-layer index the coarse
  candidates are re-scored on the fine layer with the SERVICE's rerank margin,
  the self hit is dropped by row identity, and the top-`cslsNeighbors` cosine
  similarities are averaged. The extra one-time build cost buys accurate `r(d)`
  forever.
- **no HNSW**: an exact brute-force fallback, **only for indices below 50 000
  vectors** (it is O(n²)); larger builds fail with a clear error asking for
  `hnsw: true` or a precomputed `cslsR`.
- **`cslsR` (the GPU path)**: skip the self-kNN entirely and ingest a
  precomputed `r(d)` — a float32 `.npy` of shape `(N,)`, **row-aligned with
  the input matrix** (compute it offline, e.g. batched on a GPU; remember to
  self-exclude and to average *similarities*, not distances). The values are
  validated (row count, finiteness) and stored verbatim, following the rows
  through the usual skip/dedup.

Restrictions: `csls` is **cosine-only** (the cut converts stored distances
back via `cos_sim = 1 − distance`, which only holds for the cosine metric),
and a `binary` store needs a `rerank` layer (Hamming-only distances carry no
cosine to compute the terms from).

**Saturation caveat.** After computing `r(d)` the build logs its distribution
(`csls r(d): min/p50/p95/max = …`) and **warns when the median is ≥ 0.95**:
an embedding space where every vector's neighbours sit at cosine ≈ 1 makes
`CSLS ≈ 0` for *every* candidate, so a threshold cut carries almost no signal
there — check the log line before relying on `vec:cslsThreshold`.

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
  name, each value with optional `embeddingUrl`/`embeddingModel`/`preload`
  fields:

  ```bash
  QLEVER_VECTOR_SEARCH_ENDPOINTS='{"images": {"embeddingUrl": "unix:/siglip2.private", "embeddingModel": "siglip"},
                                   "metadata": {"embeddingUrl": "unix:/qwen3.private", "preload": "lock"}}' \
    qlever-server -i myindex -p 7001
  ```

  Only the fields present are overridden (a URL-only override keeps the
  persisted model, and vice versa). The override is **in-memory only** — the
  on-disk `.meta` is never rewritten — so it is reapplied on every server
  start, and starting without the variable falls back to the persisted
  values. A malformed value is logged as a warning and ignored; it never
  prevents the server from starting.

- The same variable also configures the index's **RAM residency**: a `preload`
  field (`none` | `advise` | `lock` | `aligned`, e.g.
  `{"metadata": {"preload": "lock"}}`) selects how eagerly the flat store is
  made resident, so an operator can pin a hot index in memory — or prefault a
  cold one — **without rebuilding it**. Residency is applied when the index is
  opened at server startup (unlike the endpoint fields, it cannot change on a
  running server); the default (and an explicit `"none"`) is a plain
  demand-paged mmap.

- On a **two-layer** index (`rerank`, see above) residency is **per layer**:
  `preload` governs the coarse **scan** matrix, and a second field
  **`preloadRerank`** (same value set, default `"none"` = plain mmap)
  independently governs the fine **rerank** matrix. The headline setup pins
  the small quantized layer and leaves the big high-precision layer paged:

  ```json
  {"emb": {"preload": "lock", "preloadRerank": "none"}}
  ```

  `preloadRerank` is ignored on a single-layer index. Like `preload` it is a
  load-time serving setting; the `.meta` is never rewritten.

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
