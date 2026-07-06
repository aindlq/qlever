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

- `<name>.npy` — an `N × D` **float32, C-order** matrix (row *i* = the vector for
  entity *i*). Real embedding dims (384, 512, 768, 1024, 1536, …) are all ×64, so
  the store is SIMD-aligned for free.
- `<name>.iris` — `N` lines, **one entity IRI per line**, aligned with the matrix
  rows. Each IRI must exist in the KG (it is resolved against the vocabulary).

Produce them from any embedding pipeline, e.g. in Python:

```python
import numpy as np

iris    = ["<http://example.org/doc1>",
           "<http://example.org/doc2>",
           "<http://example.org/doc3>"]
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
| `npy`     | path to the `N × D` float32 `.npy` matrix                      |
| `iris`    | path to the row-aligned IRI list                               |
| `metric`  | `cosine` \| `dot` \| `l2` (distance; smaller = closer)         |
| `hnsw`    | `true` builds the ANN graph; `false` = exact/flat store only   |
| `scalar`  | *(optional)* storage precision `f32` (default) \| `f16` \| `bf16` \| `i8`|
| `embeddingUrl`   | *(optional)* embedding endpoint bound to this index (see below) |
| `embeddingModel` | *(optional)* model name sent to that endpoint            |

- bf16 — for bf16 embeddings: save the `.npy` as fp32 (numpy has no bf16) and
  set `scalar: bf16`; qlever stores 2-byte bf16, lossless since the fp32 is an
  upscaled bf16.

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
- Without `embeddingUrl`, the index is fully usable with explicit query
  vectors and entity↔entity distances; only `vec:embed` (and the SERVICE's
  text/image query points) error with "has no embeddingUrl configured".

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
# then rank by distance -- vec:embed returns the float-list string form
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

## 5. Re-indexing

Re-run the same `qlever-index` command whenever the KG or the vectors change.
The `.npy` bundle is the source of truth for the vectors; the KG is the source
for the entities they attach to.
