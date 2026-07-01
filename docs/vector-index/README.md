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
| `iris` | text file, one `<IRI>` per line, aligned with the vector rows | required |
| `npy` | 2-D little-endian float32 `.npy` matrix (v1/v2/v3 header) | one of `npy`/`texts` |
| `texts` | text file whose lines are embedded via `embeddingUrl` at build time | one of `npy`/`texts` |
| `dimensions` | vector dimension (inferred if omitted) | inferred |
| `metric` | `cosine`, `l2sq`, `innerProduct` | `cosine` |
| `scalar` | `f32` (f16/i8 are planned) | `f32` |
| `hnsw` | also build a usearch HNSW index | `true` |
| `hnswConnectivity` | usearch M | 16 |
| `hnswExpansionAdd` | efConstruction | 128 |
| `hnswExpansionSearch` | efSearch (applied at query time) | 64 |
| `embeddingUrl` | OpenAI-compatible embeddings endpoint bound to this index | — |
| `embeddingModel` | model name sent to the endpoint | — |

IRIs not present in the knowledge graph are skipped (with a count in the log);
duplicate IRIs are deduplicated (first vector wins, with a warning). The build
is atomic: files are written to temporaries and renamed into place (metadata
last), so an interrupted build never leaves a loadable-but-inconsistent index.

On-disk layout per index `N` of database `B` (see `VectorIndexFormat.h`):
`B.vec.N.meta` (JSON), `B.vec.N.keys` (sorted entity ids), `B.vec.N.data`
(row-major floats), `B.vec.N.hnsw` (optional usearch file). The metadata
records the **vocabulary size at build time**; a vector index whose fingerprint
does not match the loaded knowledge graph (i.e. the KG was rebuilt) is skipped
at server start with a warning instead of silently binding vectors to wrong
entities. Broken index files are likewise skipped with a warning; queries
against a skipped index fail with "no loaded vector index named ...".

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
- `vec:left ?x` — the **join form**: for each `?x` bound by a nested
  `{ ... }` pattern, emit the k nearest entities. The nested pattern must not
  bind `vec:result`/`vec:bindScore`.

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

- **Storage/serving**: the flat store is opened as a read-only mmap
  (`MmapVectorView`), the HNSW file via usearch `view()` — startup cost is
  O(metadata), the index may exceed RAM, and concurrent readers are safe. The
  configured `hnswExpansionSearch` is re-applied after `view()` (usearch does
  not persist it), and the search-context pool is enlarged beyond
  `hardware_concurrency()` so that a server running with more query threads
  than cores does not fail searches.
- **Keys are raw `ValueId` bits** (an explicitly `uint64_t`-keyed
  `index_dense_gt`), so vector results join directly with any graph data and
  exact and HNSW search share one `metric_punned_t` (comparable distances).
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

- The **builder holds all vectors in RAM** (roughly twice the raw vector data)
  and builds the HNSW single-threaded — fine up to a few million vectors;
  external-memory sorting, a parallel usearch build, merge-based vocabulary
  resolution, and batched embedding requests are the planned path to 100M+.
- The join form requires `vec:left` to be bound **inside** the SERVICE's
  nested pattern (unlike the spatial join's `<left>`); joining with the
  surrounding query and restricting its search space need the planner's
  incomplete-join machinery (follow-up).
- usearch is built with `USEARCH_USE_SIMSIMD=0` (portable scalar kernels);
  enabling SimSIMD would speed up exact search considerably.
- `f16`/`i8` storage, Parquet ingest, and migrating the six pre-existing magic
  services onto the registry are follow-ups.
