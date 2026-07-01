# Built-in vector similarity search for QLever

QLever can attach **named vector indices** to a knowledge graph, so that
approximate/exact nearest-neighbour search can be **joined with ordinary SPARQL
graph data** in a single query — e.g. *"for each green statue, find the 10 most
visually similar artworks"* against a CLIP image index, while a separate text
index serves textual-metadata similarity. The implementation uses
[usearch](https://github.com/unum-cloud/usearch) (HNSW) for approximate search
and a memory-mapped flat float store for native vector storage and exact
search.

The feature is a **drop-in magic `SERVICE` plugin**: all of its code lives in
`src/services/vectorSearch/`, wired in through the generic
`MagicServiceRegistry` (parser), `MagicServicePlannerRegistry` (planner), and
`IndexExtensionRegistry` (index build/load lifecycle) — the core parser,
planner, and index code contain no vector-specific lines. It can be disabled
entirely at configure time with `-DQLEVER_WITH_VECTOR_SEARCH=OFF`.

## Building a vector index

Vector indices are built by `qlever index` **after** the main index, from a
`.npy` float32 matrix plus a row-aligned IRI list (or from raw texts embedded
at build time):

```bash
qlever-index -i base -f kg.ttl --service-index '{
  "vectorSearch": [
    {
      "name": "clip",
      "iris": "entities.txt",
      "npy": "vectors.npy",
      "metric": "cosine",
      "hnsw": true
    }
  ]
}'
```

Per-index keys (unknown keys are rejected):

| key | meaning | default |
|---|---|---|
| `name` | index name (letters, digits, `-`, `_`) | required |
| `iris` | text file, one `<IRI>` per line, aligned with the vector rows | required except for `parquet` |
| `npy` | 2-D little-endian float32 `.npy` matrix (v1/v2/v3 header) | one of `npy`/`texts`/`parquet` |
| `texts` | text file whose lines are embedded via `embeddingUrl` at build time | |
| `parquet` | Parquet file with `uri` (string) + `embedding` (list of float32/float64) columns; carries the URIs itself, so `iris` is not needed. Requires `-DQLEVER_VECTOR_SEARCH_PARQUET=ON` (external Apache Arrow) | |
| `dimensions` | vector dimension (inferred if omitted) | inferred |
| `metric` | `cosine`, `l2sq`, `innerProduct` | `cosine` |
| `scalar` | storage type `f32`, `f16`, or `i8` (half/quarter footprint; `i8` expects inputs in [-1, 1]) | `f32` |
| `hnsw` | also build a usearch HNSW index | `true` |
| `hnswConnectivity` | usearch M | 16 |
| `hnswExpansionAdd` | efConstruction | 128 |
| `hnswExpansionSearch` | efSearch (applied at query time) | 64 |
| `buildThreads` | threads for the HNSW construction and IRI resolution | all cores |
| `embeddingUrl` | OpenAI-compatible embeddings endpoint bound to this index | — |
| `embeddingModel` | model name sent to the endpoint | — |
| `remap` | see "Re-indexing the RDF data" below | `false` |

IRIs not present in the knowledge graph are skipped (with a count in the log);
duplicate IRIs are deduplicated (first vector wins, with a warning). The build
is atomic: files are written to temporaries and renamed into place (metadata
last), so an interrupted build never leaves a loadable-but-inconsistent index.

The build is designed for very large inputs: vectors are STREAMED through
temporary files (the builder holds ~24 bytes of bookkeeping per row in RAM,
never the vectors -- 100M rows is ~2.4 GB), IRIs are resolved against the
vocabulary in parallel batches, texts are embedded in batched requests, and
the HNSW graph is constructed concurrently on `buildThreads` threads with the
vectors read from the memory-mapped store (usearch holds only the graph in
RAM, roughly `N * connectivity * 10` bytes). A concurrently built graph is not
bit-for-bit deterministic (recall is unaffected); set `buildThreads: 1` for a
deterministic build. Reference point (48-core AVX-512 Xeon, clustered data):
2M x 128-dim f32 vectors build in ~24 s with recall@10 = 1.0 and ~1.5 GB peak
builder RSS, i.e. ~83k vectors/s -- roughly 20 minutes per 100M vectors.

On-disk layout per index `N` of database `B` (see `VectorIndexFormat.h` for
details): `B.vec.N.meta` (JSON), `B.vec.N.data` (row-major floats; immutable),
`B.vec.N.iris` (row-aligned IRIs; immutable), `B.vec.N.keys` (row -> entity
id), `B.vec.N.rowmap` (entity id -> row), `B.vec.N.hnsw` (optional usearch
graph, keyed by ROW index; the vectors are NOT duplicated into it -- distances
are computed against the memory-mapped `.data` store, so each vector exists
exactly once on disk and in the page cache).

## Re-indexing the RDF data: remapping instead of rebuilding

Entity ids are vocabulary positions, which shift whenever the RDF data is
re-indexed. The metadata records the **vocabulary size at mapping time**; a
vector index whose fingerprint does not match the loaded knowledge graph is
skipped at server start with a warning (instead of silently binding vectors to
wrong entities). Broken index files are likewise skipped; queries against a
skipped index fail with "no loaded vector index named ...".

Because the vector matrix, the IRI sidecar, and the row-keyed HNSW graph never
depend on entity ids, a re-index of the RDF data does NOT require rebuilding
the vector index. Instead, run a **remap**:

```bash
qlever-index -i base -f updated-kg.ttl \
  --service-index '{"vectorSearch": [{"name": "clip", "remap": true}]}'
```

This re-resolves the persisted IRIs against the new vocabulary (in parallel)
and rewrites only the two small mapping files -- minutes and a few GB of I/O
even at the 100M-vector scale, versus hours for a full rebuild. Entities that
disappeared from the knowledge graph become tombstones: they stay in the files
but are skipped by every search. Rebuild (rather than remap) when the set of
entities with vectors has changed substantially.

## Querying

```sparql
PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>
SELECT ?painting ?score WHERE {
  SERVICE vec: {
    _:cfg vec:index "clip" ;
          vec:query <http://example.org/artwork/123> ;
          vec:result ?painting ;
          vec:bindScore ?score ;
          vec:k 10 .
  }
  ?painting <is-a> <Painting> .
}
```

Exactly one **query point** must be given:

- `vec:queryVector "0.1,0.2,..."` — an explicit vector;
- `vec:query <iri>` — the stored vector of a constant entity;
- `vec:queryText "..."` — free text, embedded at query time via the index's
  `embeddingUrl` (over HTTP(S), a `unix:/path.sock` socket, or in-process
  `llama:/model.gguf` when built with `-DQLEVER_WITH_LLAMACPP=ON`);
- `vec:imageUrl <...>` / `vec:imageBase64 "..."` — an image, embedded at query
  time (server-local file paths are deliberately not supported: that would let
  remote clients read arbitrary server files);
- `vec:left ?x` — the **join form**: for each `?x`, emit the k nearest
  entities. `?x` is bound either by a nested `{ ... }` pattern inside the
  SERVICE, or — like the spatial join — by the **surrounding query**:

  ```sparql
  ?x <is-a> <Painting> .
  SERVICE vec: { _:c vec:index "clip" ; vec:left ?x ;
                 vec:result ?nn ; vec:k 10 . }
  ```

  (The `<left>` variable must then be the only variable shared with the
  service clause, and the nested pattern must not bind
  `vec:result`/`vec:bindScore`.)

Further parameters: `vec:result`/`vec:right` (required; the result entity
variable), `vec:bindScore` (optional distance variable, smaller = more
similar), `vec:k`/`vec:numNearestNeighbors` (default 10),
`vec:maxDistance` (optional distance cutoff), and `vec:algorithm`
(`vec:exact`, `vec:hnsw`, or `vec:auto` = HNSW if available).

For the point-query forms, an optional nested `{ ... }` pattern that binds the
result variable **restricts the search space** to exactly those candidate
entities (always exact search — right for selective candidate sets); the
pattern must bind only the result variable. An empty candidate set yields an
empty result. Results computed through an external embedding endpoint
(`queryText`/image) are never cached, since the endpoint's model can change.

## Design notes

- **Storage/serving**: everything is opened as read-only mmaps; the HNSW
  graph via usearch `view()` — startup cost is O(metadata), the index may
  exceed RAM, and concurrent readers are safe. Searches use a pooled set of
  usearch contexts; a burst beyond the pool briefly queues instead of failing.
- **The HNSW graph is usearch's low-level `index_gt`, keyed by row index**,
  with a custom metric that reads vectors from the flat store — this is what
  keeps vectors out of the graph file and makes remapping possible. Search
  results are translated rows -> current entity ids via `.keys`. Exact and
  HNSW search share one `metric_punned_t` (comparable distances). The exact
  search switches between per-candidate lookups and a sequential
  scan-with-filter depending on candidate density.
- SIMD kernels (NumKong, usearch's SimSIMD successor) are ON by default
  (`-DQLEVER_VECTOR_SEARCH_SIMD=OFF` to disable). The per-ISA kernels
  (AVX2/AVX-512/AMX on x86) are enabled by compiler probes and selected per
  CPU at run time -- no `-march` change, the binary stays portable. Beware
  when integrating usearch/NumKong elsewhere: without the probe-driven
  `NK_TARGET_*` defines it silently compiles serial-only kernels.
- **Operations**: `VectorSearch` (point query, optional candidate restriction)
  and `VectorSearchJoin` (per-row form, memoizes repeated query entities)
  follow the `SpatialJoin` conventions — allocator-backed `IdTable`s,
  cancellation checks inside the scan loops, cache keys covering every
  semantic ingredient (including the candidate column and bit-exact query
  vectors), cost estimates that reflect exact-scan vs. HNSW-probe cost.
- **Extensibility**: the registries make a magic service a self-contained
  folder; services are OBJECT libraries linked into `qlever-server`,
  `qlever-index`, `LibQLeverExample`, and the test binary, so the
  self-registration cannot be dead-stripped. External libqlever embedders must
  link the service object libraries themselves.

## Known limitations / follow-ups

- For a fast HNSW build, the flat store should fit in the page cache (graph
  construction reads vectors in random order); with data much larger than
  RAM, expect the build to become NVMe-bound. `f16`/`i8` storage halves or
  quarters that requirement — and is also FASTER at query time on modern CPUs
  (memory-bandwidth-bound workload with FP16/VNNI kernels): at 500k x 128 on
  a 48-core AVX-512 Xeon, queries took 1.6 s (f32) / 1.0 s (f16) / 0.5 s (i8)
  at recall@10 0.98–1.0. The graph itself always stays small.
- The outer-bound `vec:left` form searches the whole index per row; combining
  it with a candidate restriction (an additional nested pattern) is a
  follow-up.
- Tombstones accumulate over repeated remaps; rebuild when a large fraction of
  the indexed entities has disappeared (searches over-fetch past tombstones).
- Migrating the six pre-existing magic services onto the registry is a
  follow-up.
