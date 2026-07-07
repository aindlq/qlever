# Vector search in QLever — building, querying, and how it works

Four parts: (1) how you **build** an index; (2) the **runtime** (server-start)
config; (3) how you **query** it from SPARQL; (4) how it **works** inside. A
short final section contrasts the design with PR #3043. See also
[`indexing.md`](indexing.md) and, for the rationale,
[`index-payload-design.md`](index-payload-design.md).

## Part 1 — Building an index

Vectors are payload, not RDF, so you don't write them as triples. You hand
`qlever-index` a **`.npy` matrix + a list of entity IRIs**, plus a small config
in the `--service-index` JSON (`{"vectorSearch": [ … ]}`; each array entry
builds one named index):

```bash
qlever-index -i myindex -f kg.ttl -F ttl \
  --service-index '{"vectorSearch":[
     {"name":"emb","npy":"emb.npy","iris":"emb.iris",
      "metric":"dot","scalar":"bf16","hnsw":true}
  ]}'
```

This builds the normal QLever index from `kg.ttl`, then the vector index `emb`.
It's reachable in queries as `vidx:emb` (= `<…/vectorSearch/index/emb>`), the
same IRI you introspect.

### Input

- **`npy`** — an `N × D` matrix `.npy`, **float32** (`<f4`) or **ml_dtypes
  bfloat16** (`<V2`), C-order; row *i* is entity *i*'s vector. (Near-zero-copy
  for the ×64 dims real models use.)
- **`iris`** — one **entity IRI per line** (bare `http://…/doc1` or bracketed
  `<http://…/doc1>`), row-aligned with the matrix; each IRI must already exist in
  the KG (otherwise that row is skipped with a warning).

Vectors are **precomputed**: embed your corpus offline with any pipeline and
`np.save` an `fp32` or `ml_dtypes` `bf16` array. (Embedding a *query* at query
time is a separate, runtime-configured concern — Part 2 — not a build input.)

### Settings keys

| key | default | meaning |
|---|---|---|
| `name` | — | index name → `vidx:<name>` in queries |
| `npy` | — | the `.npy` matrix (the vectors) |
| `iris` | — | row-aligned entity-IRI list |
| `metric` | `cosine` | `cosine` \| `dot` \| `l2` — the space's metric (distance is smaller-is-closer) |
| `scalar` | `f32` | storage precision `f32` \| `f16` \| `bf16` \| `i8`, **independent** of the `.npy` dtype (an `<f4` file + `scalar:"bf16"` stores 2-byte bf16) |
| `dimensions` | *(inferred)* | if set, cross-checked against the `.npy` shape → **hard error** on mismatch |
| `hnsw` | `true` | build the HNSW graph (required for the `SERVICE`; exact brute force works without it) |
| `hnswConnectivity` | `16` | HNSW graph degree (M) |
| `hnswExpansionAdd` | `128` | HNSW build-time `ef` |
| `hnswExpansionSearch` | `64` | HNSW query-time `ef` (recall vs. latency) |
| `buildThreads` | `0` (auto) | index-build parallelism |
| `remap` | `false` | on KG rebuild, re-resolve IRIs + rewrite only the id sidecar (keep the matrix) |

For **normalized** embeddings (SigLIP2, Qwen3): use `metric:"dot"` (cosine ≡ dot
on unit vectors, and dot skips the norm) and `scalar:"bf16"` (half the RAM,
lossless from the fp32/bf16 input).

## Part 2 — Runtime (server-start) configuration

Two things are **serving** concerns, not index data — the query-time embedding
endpoint and RAM residency — so they are set at **server start**, never at build
and never persisted in the index. They come from one environment variable,
`QLEVER_VECTOR_SEARCH_ENDPOINTS`, a JSON object keyed by index name:

```yaml
# docker-compose
environment:
  QLEVER_VECTOR_SEARCH_ENDPOINTS: >
    {"emb":    {"embeddingUrl":"unix:/qwen3.private","embeddingModel":"qwen3","preload":"lock"},
     "images": {"embeddingUrl":"http://siglip:8000","embeddingModel":"siglip","preload":"aligned"}}
```

| field | meaning |
|---|---|
| `embeddingUrl` | query-time embed endpoint for `vec:embed`: `http(s)://host:port` **or** `unix:/path/to.sock` (the client POSTs to `/v1/embeddings`) |
| `embeddingModel` | model name sent to the endpoint; also the **model identity** stamped into the typed query literal (comparability) |
| `preload` | RAM residency: `none` (mmap) \| `advise` (madvise) \| `lock` (mlock) \| `aligned` (huge-page copy) — pin a hot, fits-in-RAM index |

To move an endpoint or lock an index: edit the env and **restart** the server —
no rebuild. `preload:"lock"`/`"aligned"` on a large index needs the container
allowed to lock that much memory (`ulimits: memlock: -1`, or `cap_add:
[IPC_LOCK]`). An index with no `embeddingUrl` still answers entity / inline /
`vec:vector` queries — only `vec:embed` needs the endpoint.

## Part 3 — Querying from SPARQL

Every query uses these prefixes:

```sparql
PREFIX vec:  <https://qlever.cs.uni-freiburg.de/vectorSearch/>
PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>
```

An **index** is referenced by its IRI `vidx:<name>` — the *same* IRI you
introspect and the one you search. The three functions, plus one magic predicate:

| function | meaning |
|---|---|
| `vec:distance(vidx:X, S1, S2)` | uniform **smaller-is-closer** distance between two *sources*; each source is an entity (→ its stored vector) or a query vector |
| `vec:embed(vidx:X, input)` | embed text / an image IRI via index X's endpoint → a query vector |
| `vec:vector(vidx:X, ?e)` | entity `?e`'s stored vector *from index X* as a query vector |
| `vidx:X vec:hasMember ?e` | **magic predicate**: binds `?e` to the entities that have a vector in index X (merge-join; see below) |

The metric (cosine / dot / l2) is a property of the index, so there is **one**
distance function — you never pass a metric, and every ranking is `ORDER BY ?d ASC`.

**What `vec:embed` and `vec:vector` return** — a **typed float-list literal**,
the value `vec:distance` consumes (they are never usually projected, but if you
`SELECT` one you'd see):

```
vec:embed(vidx:emb, "a red bicycle")  →  "0.13,-0.02, … ,0.08"^^<https://qlever.cs.uni-freiburg.de/vectorSearch/vec/qwen3/bf16>
vec:vector(vidx:emb, <doc/42>)        →  "0.51,0.09, … ,-0.11"^^<https://qlever.cs.uni-freiburg.de/vectorSearch/vec/qwen3/bf16>
```

The datatype `…/vec/MODEL/PRECISION` (MODEL = the index's `embeddingModel`,
PRECISION = its `scalar`) travels with the value, so `vec:distance` can check
that a query vector belongs to the space it's comparing against.

### Membership — `vidx:X vec:hasMember ?e`

`vidx:X vec:hasMember ?e` enumerates exactly the entities that have a vector in
index X. It is **cheap**: the index physically stores its entities as a
`ValueId`-sorted key list (tombstones excluded), so this scans one, already
`ValueId`-sorted, single column — no vectors are materialised — and the planner
**merge-joins** it with the rest of your query. Two uses:

- **Drop `FILTER(BOUND)`.** Instead of enumerating a broader set and pruning the
  non-members by the `vec:distance → UNDEF` sentinel, join to the members up
  front — `?e` is then guaranteed a member:
  ```sparql
  SELECT ?doc ?d WHERE {
    ?doc a :Document .
    vidx:emb vec:hasMember ?doc .                                # members only
    BIND(vec:distance(vidx:emb, ?doc, vec:embed(vidx:emb, "a red bicycle")) AS ?d)
  } ORDER BY ?d LIMIT 10                                         # no FILTER(BOUND) needed
  ```
- **Enumerate the whole index.** `vidx:X vec:hasMember ?e` alone yields exactly
  the whole index — which is the WHOLE-INDEX input the HNSW fast path (`useIndex`
  / the 4th `vec:distance` argument, use case 6) requires to engage.

### Use case 1 — semantic search by text
```sparql
SELECT ?doc ?d WHERE {
  ?doc a :Document .
  BIND(vec:distance(vidx:emb, ?doc, vec:embed(vidx:emb, "a red bicycle")) AS ?d)
  FILTER(BOUND(?d))                      # entities without a vector -> UNDEF, dropped
} ORDER BY ?d LIMIT 10
```
`vec:embed` embeds the text once (via the index's endpoint), and `vec:distance`
looks up each `?doc`'s stored vector. A selective pattern (`?doc a :Document`)
keeps this exact and index-free.

### Use case 2 — search by an image (cross-modal, e.g. SigLIP)
```sparql
SELECT ?img ?d WHERE {
  ?img a :Image .
  BIND(vec:distance(vidx:img, ?img, vec:embed(vidx:img, <http://example.org/query.jpg>)) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 10
```
An **IRI** argument to `vec:embed` is embedded as an image (the endpoint must be
a multimodal/vision embedding server). Text→image works when the model aligns
both in one space.

### Use case 3 — "find things similar to *this* entity"
```sparql
# to one fixed entity:
SELECT (vec:distance(vidx:img, <img/mona-lisa>, <img/starry-night>) AS ?d) {}

# every pair, nearest first — both sources are entities (stored-vector lookups):
SELECT ?a ?b ?d WHERE {
  ?a a :Image . ?b a :Image . FILTER(str(?a) < str(?b))
  BIND(vec:distance(vidx:img, ?a, ?b) AS ?d)
} ORDER BY ?d
```
No embedding, no query vector — both vectors are already in the index.

### Use case 4 — a precomputed query vector (inline, typed)
```sparql
BIND(vec:distance(vidx:emb, ?e,
     "0.12,-0.03, … ,0.44"^^<https://qlever.cs.uni-freiburg.de/vectorSearch/vec/qwen3/bf16>) AS ?d)
```
Give the vector as a comma-separated float string **carrying the space datatype**
`…/vec/MODEL/PRECISION` — the same shape `vec:embed`/`vec:vector` return.
`vec:distance` validates the datatype's **model + precision** and the **float
count (dimension)** against the index; a mismatch is UNDEF, not a wrong number.
(A bare float string with no datatype is still accepted, but then only the
dimension is checked — you lose the model/precision guard, so prefer the typed
form.)

### Use case 5 — cross-index (two indices, same model & precision)
```sparql
SELECT ?a ?p ?d WHERE {
  ?a a :Artwork . ?p a :Photo .
  BIND(vec:distance(vidx:artwork, ?a, vec:vector(vidx:photo, ?p)) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d
```
`vec:vector(vidx:photo, ?p)` pulls `?p`'s vector out of the *photo* index tagged
with its model+precision; `vec:distance` checks that against `vidx:artwork`
(**model + precision + dimension**) → matches compute, a mismatch is UNDEF.
(Simplest alternative: if both are the same model, put them in **one** index and
`vec:distance(vidx:visual, ?a, ?p)`.)

### Use case 6 — whole-index top-k, exact (no SERVICE)
```sparql
SELECT ?doc ?d WHERE {
  ?doc <hasEmbedding> ?anything .                     # any pattern that enumerates the members
  BIND(vec:distance(vidx:emb, ?doc, vec:embed(vidx:emb, "a red bicycle")) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 10
```
Search is just `ORDER BY distance LIMIT k` — exact, brute-force, parallel across
cores (Part 4). Fine into the millions of vectors.

**Optional HNSW acceleration (approximate).** A constant non-negative integer
fourth argument, `vec:distance(vidx:emb, ?doc, Q, k')`, opts the *same* query
into the index's HNSW graph: it runs one whole-index `searchHnsw(Q, k')` and
binds a distance only for those ~`k'` nearest entities, so the usual
`FILTER(BOUND(?d)) ORDER BY ?d LIMIT n` then sorts only ~`k'` rows instead of the
whole column. This is **guarded to inputs that cover the index** — it is taken
only when the input covers the entire live set (its row count is *at least* the
index's vector count) and one side is the constant query `Q`. Covering inputs
include the exact whole-index scan *and supersets of it* — e.g. all works of a
type where some works carry no vector — because the HNSW top-`k'` entities are
then all present in the input, so the result is complete. On a *selective*
filter (fewer input rows than index vectors) or an index without an HNSW graph
it **silently falls back to the exact path**, because HNSW searches the whole
graph and could otherwise miss in-subset entities. It is
**approximate** (bounded by HNSW recall): larger `k'` trades speed for recall,
`k' = 0` (or omitting it) is exact. Use it for a huge *unfiltered* top-k where
exact brute force is too slow; keep the exact form when the enumerator already
selects a small subset.

### Use case 7 — whole-index top-k, accelerated (HNSW `SERVICE`)
For a huge *unfiltered* top-k, the opt-in HNSW graph via the `SERVICE` block
(exact never needs it; HNSW is always explicit). See `indexing.md` for the shape.

### Use case 8 — introspect an index (it's a real RDF resource)
```sparql
SELECT ?model ?dim ?metric ?count WHERE {
  vidx:emb  vec:model ?model ; vec:dimension ?dim ; vec:metric ?metric ; vec:count ?count .
}
```

## Part 4 — How it works

The *decision* (a vector is searchable **payload**, not RDF) and its rationale
live in [`index-payload-design.md`](index-payload-design.md); this is the
mechanism, concisely.

- **Storage.** Each index is one contiguous, fixed-stride, **bytes-only** matrix
  for a single `(model, dim, precision)` space, `mmap`'d — vectors never enter
  the vocabulary, permutations, or any triple. Rows are stored in **entity-id
  order** (the builder sorts by entity id and gathers the matrix in that order).
  Two small sidecars bridge the two directions: `keys` (row → id, monotonic) and
  `rowmap` (id → row, id-sorted for `lower_bound`). *Why id-sorted:* a
  filtered/whole-index scan then reads the matrix **sequentially**
  (prefetch-friendly), the result stays merge-joinable with QLever's sorted
  world, and the same matrix backs the HNSW graph.

- **Precision & SIMD.** The stored scalar (`f32`/`f16`/`bf16`/`i8`) is used
  **as-is** — no up-conversion. Distances run through NumKong kernels that
  runtime-dispatch to the widest ISA the CPU has (AVX-512 FP16/BF16, AMX, NEON,
  …), reported once at startup (`Vector search SIMD: …`). *Why:* a brute-force
  scan is memory-bandwidth-bound, so storing at the target precision (`bf16` = 2
  bytes) directly halves the bytes read; decoding to fp32 first would throw that
  away.

- **One distance, exact + ANN.** A single punned metric backs both the exact
  scan and the HNSW graph, so their distances are identical. Every metric is
  normalized to **smaller-is-closer** (`cosine → 1−cos`, `dot → 1−dot`, `l2 →
  distance`).

- **Brute-force query.** `ORDER BY vec:distance(...) LIMIT k` is exact top-k: it
  scores **every** candidate (the `LIMIT` trims the output, not the work), then
  sorts. The scan is **parallelized across cores** (OpenMP; bound by the
  *aggregate* memory bandwidth). When the per-row entity column is the query
  result's **leading sort key** — the common `?e … BIND(vec:distance)` after a
  join — it takes a **merge-walk** fast path: one `lower_bound` per chunk then a
  linear walk of the id-sorted rowmap instead of a binary search per row, and
  **consecutive duplicate entities are scored once**. Sub-linear HNSW top-k is
  opt-in via the `SERVICE` only; the planner never silently rewrites an exact
  `ORDER BY … LIMIT` into an ANN lookup.

- **Query values are transient.** `vec:embed` (text/image → endpoint) and
  `vec:vector` (an entity's stored vector) both yield a **typed float-list
  literal** `"…"^^<…/vec/model/precision>` — a value that lives only for the
  query, never a stored term. `vec:distance` validates that datatype's model +
  precision + dimension against the index it's called on and returns UNDEF on a
  mismatch, which is what makes cross-index distance safe.

- **Load & config.** At startup the loader auto-discovers each
  `‹basename›.vec.‹name›.meta`, `mmap`s the matrix + HNSW, materializes the
  index's metadata triples (`vec:model/dimension/metric/count`) as delta triples
  on the `vidx:‹name›` resource, and applies the `QLEVER_VECTOR_SEARCH_ENDPOINTS`
  env (endpoint + residency) in memory. Nothing about serving is baked into the
  index — no re-passing of `--service-index` to `qlever-server`.

## Part 5 — How the design differs from PR #3043

Both projects reached the same *native-first* instinct — a vector decoded to a
compact sidecar, not re-parsed per query, and the embedding "type" as a queryable
RDF resource. The divergence is **what the vector *is***.

| | **PR #3043** | **Ours (index-payload)** |
|---|---|---|
| A vector is… | a first-class **RDF literal** `"[…]"^^emb:fp32Vector` in the vocabulary | **index payload** — bytes in a per-space matrix, never an RDF term |
| Per-embedding RDF | reified `<e> emb:hasEmbedding _:b . _:b emb:asFp32Vector … ; emb:type …` | **none** (metadata is once **per index**, not per embedding) |
| Precision | fp32 | f32 / f16 / **bf16** / i8, stored + scanned as-is |
| Query vector | a materialised RDF literal you pass in | a **transient** typed value from `vec:embed`/`vec:vector`/inline |
| Query-time embedding | none — precompute externally | **`vec:embed`** (text/image; http or `unix:` socket) |
| ANN | none — brute-force only | optional **HNSW** via `SERVICE` (exact is the default) |
| Comparability | pass the `type` IRI; hard error on mismatch | model+precision in the typed literal, validated → **UNDEF** on mismatch |

**Why not RDF literals** (we built and dropped exactly #3043's choice): there is
no real `SELECT`/`CONSTRUCT` use for a raw vector (entity↔entity passes the
*entity*; bulk export wants binary, not float-text); the literal is
full-precision text regardless of storage precision, so an `i8` vector carries a
~8× text twin and quantisation buys nothing; and per-embedding reification is ~3
triples + a blank node × six permutations ≈ **1.8 B entries at 100 M vectors**
versus **zero** for us. Keeping the vector out of RDF is what enables the
compact, id-sorted, per-precision SIMD matrix — and with it the sequential scan,
the parallel + merge-walk query path, query-time embedding, and the ANN option.
