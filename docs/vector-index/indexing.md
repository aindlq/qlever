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
| `name`    | index name; used in `vec:distance("name", …)` and the SERVICE  |
| `npy`     | path to the `N × D` float32 `.npy` matrix                      |
| `iris`    | path to the row-aligned IRI list                               |
| `metric`  | `cosine` \| `dot` \| `l2` (distance; smaller = closer)         |
| `hnsw`    | `true` builds the ANN graph; `false` = exact/flat store only   |
| `scalar`  | *(optional)* storage precision `f32` (default) \| `f16` \| `bf16` \| `i8`|

- bf16 — for bf16 embeddings: save the `.npy` as fp32 (numpy has no bf16) and
  set `scalar: bf16`; qlever stores 2-byte bf16, lossless since the fp32 is an
  upscaled bf16.

## 4. Serve and query

```bash
qlever-server -i myindex -p 7001 &
```

Rank entities by distance to a query vector — plain SPARQL, exact, no SERVICE:

```sparql
PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>
SELECT ?doc ?dist WHERE {
  ?doc a <http://example.org/Document> .
  BIND(vec:distance("emb", ?doc, "0.12,-0.03, … ,0.44") AS ?dist)   # a query vector, D comma-separated floats
  FILTER(BOUND(?dist))                                              # drop entities with no vector
}
ORDER BY ?dist
LIMIT 10
```

- `vec:distance("emb", ?entity, "<floats>")` — query point is a comma-separated
  float list; you can also pass a **constant entity IRI** to use its stored
  vector ("find documents similar to `<doc1>`").
- The metric is a property of the index, so distances are uniformly
  smaller-is-closer → always `ORDER BY ?dist ASC`.
- For a huge, *unfiltered* whole-index top-k, use the HNSW `SERVICE` instead of
  the brute-force scan above (see the design doc). A selective filter on the
  entity (like `?doc a :Document`) keeps the exact brute-force path fast.

## 5. Re-indexing

Re-run the same `qlever-index` command whenever the KG or the vectors change.
The `.npy` bundle is the source of truth for the vectors; the KG is the source
for the entities they attach to.
