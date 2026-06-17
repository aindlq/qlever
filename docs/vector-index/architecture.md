# Vector similarity search in QLever — architecture & extension design

> Status: design proposal. Companion documents: [`integration-plan.md`](./integration-plan.md)
> (phased implementation plan) and [`README.md`](./README.md) (overview).

## 1. Goal

Make QLever able to host one or more **built-in vector indices** (e.g. a CLIP
index over artwork images and a Qwen3 index over textual metadata), and to
**join vector similarity results with normal SPARQL graph data** inside a single
query.

Concretely we want a query like *"take all green statues, then for each find the
10 most visually similar artworks"* to run as one plan, where the "green statues"
part is ordinary SPARQL and the "most similar" part is a k-nearest-neighbour
(kNN) search against a CLIP vector index.

Design requirements taken from the brief:

- **Multiple, named vector indices** per database (different models / modalities /
  embedding spaces), each with its own dimensionality and metric.
- **Indices are optional.** A small database can run **exact** (brute-force)
  similarity search with no ANN index at all. A large database can additionally
  build an approximate (HNSW) index for scale.
- **Queries can request a specific index** and ask for exact or approximate
  search.
- **Native, cache-friendly vector storage** — a contiguous, memory-mapped float
  store so we can work with indices far larger than RAM, analogous to how
  geometries are stored today.
- **Bulk input** from columnar files (`(URI, vector)` pairs) — Parquet and/or
  `.npy`/`.npz`.
- **Cardinality-aware optimisation**: when one side of a similarity join is
  already small (e.g. 5 000 green statues), run exact search over that small set
  instead of probing the ANN index.

This is deliberately modelled on QLever's existing **spatial join**, which is the
same shape of problem (nearest-neighbour / range search on a positional column,
joined with the rest of the query). Wherever possible we mirror its design so the
feature is idiomatic and maintainable.

## 2. What we build on (three existing patterns)

QLever already contains every architectural ingredient we need. The vector index
is a recombination of three existing features:

| Concern | Existing analogue | Key files |
| --- | --- | --- |
| A magic `SERVICE` that does kNN / range search and joins with the query, chosen by the planner with exact vs. indexed algorithms | **Spatial join** | `src/engine/SpatialJoin.{h,cpp}`, `src/engine/SpatialJoinAlgorithms.{h,cpp}`, `src/engine/SpatialJoinConfig.h`, `src/parser/SpatialQuery.{h,cpp}` |
| Compact, on-disk, per-entity payload stored separately from the main vocabulary, with precomputed fixed-stride records | **Geo vocabulary / GeometryInfo** | `src/index/vocabulary/GeoVocabulary.{h,cpp}`, `src/index/vocabulary/SplitVocabulary*.h`, `src/rdfTypes/GeometryInfo.h` |
| An **optional** auxiliary index, built in a separate pass from separate input files, persisted as extra files, loaded on demand at startup, and exposed to operations through the `Index` object | **Text index** | `src/index/TextIndexBuilder.{h,cpp}`, `src/index/IndexImpl.Text.cpp`, `src/engine/TextIndexScanForEntity.cpp` |

The vector index borrows its **query/operation/planner** structure from the
spatial join, its **on-disk storage layout** from the geo vocabulary, and its
**build/load lifecycle** from the text index.

A fourth ingredient — the dependency — is brought in exactly like the existing
third-party C++ libraries (see §10).

## 3. Why vectors are stored out-of-line (not in a `ValueId`)

QLever encodes every value as a 64-bit `ValueId`: 4 datatype bits + 60 payload
bits (`src/global/ValueId.h:103-111`). "Folded" datatypes such as `GeoPoint`
pack their whole value into those 60 bits — a `GeoPoint` is two 30-bit reduced
coordinates (`src/rdfTypes/GeoPoint.h:52-62`). That is the ceiling: **60 bits
total**, i.e. at most one 32-bit float.

A real embedding is hundreds to thousands of 32-bit floats, so it **cannot** be
folded into a `ValueId`. Vectors must live in an out-of-line store and be
referenced by an index. This is the same conclusion that led geometries to be
stored in a `GeoVocabulary` side file rather than inside the `Id`.

We therefore do **not** put vectors in the triple store or the vocabulary at all.
A vector index is an **auxiliary index attached to existing entities**: it maps an
entity's `VocabIndex` to its dense vector. (Optionally, a thin
`Datatype::VectorIndex` tag can later be added to *reference* a stored vector as a
SPARQL value; this is not required for the core join feature — see §11.)

## 4. Data model and SPARQL surface

### 4.1 Model

A vector index `N` is a function `entity → vector ∈ ℝ^D_N` defined for a subset of
the database's entities, together with a metric `m_N` (cosine / L2² / inner
product). Multiple indices coexist; each has its own `D` and `m`.

The core query primitive is a **kNN similarity join**, structurally identical to
the spatial nearest-neighbour join:

> For each binding of a *query* variable, find the `k` *result* entities whose
> vectors (in a chosen index) are nearest to the query vector, and join them in,
> optionally binding the similarity score.

The query vector can be supplied two ways:

- **By entity** — `?queryEntity` binds to an entity that itself has a vector in
  the index (look the vector up). This expresses "similar to *these* things".
- **By literal** — an inline query vector, for "similar to this externally
  supplied embedding" (e.g. a CLIP vector computed from an uploaded image).

### 4.2 Magic SERVICE syntax

We add a magic service IRI, mirroring `SPATIAL_SEARCH_IRI`
(`src/parser/MagicServiceIriConstants.h:18-28`):

```
constexpr inline std::string_view VECTOR_SEARCH_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/>";
```

**Example A — "for each green statue, the 10 most visually similar artworks":**

```sparql
PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>
SELECT ?statue ?similar ?score WHERE {
  # ordinary SPARQL: the (small) left/query set
  ?statue a :Statue ; :color :green .

  SERVICE vec: {
    _:cfg vec:index     "clip" ;       # which named vector index
          vec:left      ?statue ;      # query entities (vector looked up in "clip")
          vec:right     ?similar ;     # result entities to bind
          vec:k         10 ;           # number of neighbours per query entity
          vec:metric    vec:cosine ;   # optional; default = index's metric
          vec:bindScore ?score .       # optional output column
    # optional nested pattern restricting the search space (the "right" side):
    { ?similar a :Painting . }
  }
}
```

If the nested `{ ... }` pattern is omitted, the search space is the **entire**
`clip` index (the natural case for HNSW). If present, it restricts results to the
matched entities (and enables the exact-over-small-set optimisation, §8–9).

**Example B — search by an externally supplied vector:**

```sparql
SELECT ?img ?score WHERE {
  SERVICE vec: {
    _:cfg vec:index       "clip" ;
          vec:queryVector  "0.013,-0.21,...,0.07"^^vec:f32 ;  # inline query
          vec:right        ?img ;
          vec:k            20 ;
          vec:bindScore    ?score .
  }
}
```

**Example C — force exact search regardless of size** (e.g. to validate recall):

```sparql
    _:cfg vec:index "clip" ; vec:left ?a ; vec:right ?b ;
          vec:k 5 ; vec:algorithm vec:exact .
```

### 4.3 Config parameters

Parsed by a new `parsedQuery::VectorSearchQuery : MagicServiceQuery`, mirroring
`SpatialQuery` (`src/parser/SpatialQuery.cpp:23-126`). Each predicate sets one
optional field; unknown predicates throw at parse time.

| Predicate | Field | Notes |
| --- | --- | --- |
| `vec:index` | index name (string) | required; must match a loaded index |
| `vec:left` | query entity variable | mutually exclusive with `vec:queryVector` |
| `vec:queryVector` | inline vector literal | mutually exclusive with `vec:left` |
| `vec:right` | result entity variable | required |
| `vec:k` | `maxResults_` (size_t) | kNN count |
| `vec:maxDistance` / `vec:minScore` | distance/score threshold | optional range cap |
| `vec:bindScore` | score output variable | optional |
| `vec:metric` | `vec:cosine` / `vec:l2sq` / `vec:innerProduct` | optional; defaults to the index's build metric |
| `vec:algorithm` | `vec:exact` / `vec:hnsw` / `vec:auto` | optional; default `vec:auto` (§9) |
| `vec:payload` | extra variables to carry through | as in spatial join's `<payload>` |

A `VectorSearchQuery::toVectorJoinConfiguration()` method validates cross-field
constraints (e.g. `left`/`queryVector` exactly one; `index` known; metric
compatible with the index) and lowers to a dependency-light
`VectorJoinConfiguration` struct, exactly as
`SpatialQuery::toSpatialJoinConfiguration()` does
(`src/parser/SpatialQuery.cpp:129-230`).

## 5. Native vector storage

### 5.1 Layout — one self-describing store per named index

Each named index `N` produces a small set of files alongside the main index,
following QLever's `base + suffix` convention (cf. `VOCAB_SUFFIX` etc. in
`src/global/Constants.h:226-228`, and `base.text.*` for the text index):

```
<base>.vec.<N>.meta      # JSON: dim D, metric, scalar type, count, hasHnsw, build params
<base>.vec.<N>.keys      # MmapVector<VocabIndex>, ascending  (row i -> entity id)
<base>.vec.<N>.data      # flat row-major float store, stride = D * sizeof(scalar)
<base>.vec.<N>.hnsw      # usearch index file (optional; present iff hasHnsw)
```

The `.keys` + `.data` pair is the **native store**: a contiguous, fixed-stride
matrix that is *exactly* the structure already used for `GeometryInfo` records
(`src/index/vocabulary/GeoVocabulary.h:42-48`), but mapped with
`ad_utility::MmapVectorView<T>` instead of per-record `pread`, giving zero-copy,
page-cache-backed random access (`src/util/MmapVector.h:334-401`).

- **Keying.** Row `i` of `.data` belongs to entity `keys[i]` (a `VocabIndex`).
  `keys` is sorted ascending, so `entity → row` is a binary search and
  `row → entity` is `O(1)`. The vectors are stored densely (only entities that
  *have* a vector occupy a row), so a sparse index over a large database costs
  only `count · (8 + D·sizeof(scalar))` bytes.
- **Memory mapping.** At query time both files are opened read-only with
  `MmapVectorView` and `AccessPattern::Random` (forwarded to `madvise`,
  `src/util/MmapVector.h:118,270`). This is the "huge index, memory-mapped file"
  requirement: only touched pages are paged in.
- **Scalar type / quantisation.** `scalar ∈ {f32, f16, i8, b1}` is recorded in
  `.meta`. f16/i8/b1 cut the store (and HNSW) size 2×/4×/32× at some recall cost;
  usearch supports all of them natively. Default `f32`.

### 5.2 Multiple indices and the manifest

The set of vector indices is recorded as one key in the existing index manifest
`*.meta-data.json` (written/read by `IndexImpl::writeConfiguration()` /
`readConfiguration()`, `src/index/IndexImpl.cpp:1173-1234`):

```json
"vector-indices": {
  "clip":  { "dim": 768,  "metric": "cosine", "scalar": "f32", "count": 12345678, "hasHnsw": true  },
  "qwen3": { "dim": 1024, "metric": "cosine", "scalar": "f16", "count": 9876543,  "hasHnsw": false }
}
```

Because vector indices are **additive and optional**, their presence is detected
from this manifest key plus file existence — **no bump of the global
`indexFormatVersion`** is required (`src/index/IndexFormatVersion.h:42`). Bumping
that constant would force every existing QLever index in the world to be rebuilt;
we avoid it. (We *do* keep a small independent `VECTOR_INDEX_VERSION` in each
`.vec.<N>.meta` so a future layout change can invalidate just the vector files,
exactly like `GEOMETRY_INFO_VERSION`, `src/rdfTypes/GeometryInfo.h:163-171`.)

### 5.3 The runtime accessor: `VectorIndex`

A new class `VectorIndex` (in `src/index/vectorIndex/`) owns, per named index:
the mmap'd `keys`/`data` views, the parsed `.meta`, and an optional loaded
usearch index (`view()`-mapped). Its API is the storage-and-search primitive the
engine calls:

```cpp
class VectorIndex {
 public:
  size_t dim() const;
  VectorMetric metric() const;
  bool hasHnsw() const;

  // entity -> vector (nullptr if this entity has no vector in this index)
  std::optional<std::span<const float>> getVector(VocabIndex id) const;

  // Exact brute force over an explicit candidate set (the small-set path, §8).
  // Returns the top-k (entity, distance) for one query vector.
  std::vector<ScoredId> searchExact(std::span<const float> query, size_t k,
                                    std::span<const VocabIndex> candidates,
                                    std::optional<float> maxDist) const;

  // Approximate HNSW search over the whole index, with an optional predicate
  // filter (usearch supports a per-candidate filter callback).
  std::vector<ScoredId> searchHnsw(std::span<const float> query, size_t k,
                                   std::optional<EntityFilter> filter,
                                   std::optional<float> maxDist) const;
};
```

A registry of `name → VectorIndex` hangs off `IndexImpl` next to `textMeta_` /
`docsDB_` (`src/index/IndexImpl.h:126-129`) and is reached from operations via
`getExecutionContext()->getIndex()` — the same path the text scans use
(`src/engine/Operation.h:163`, `TextIndexScanForEntity.cpp:34`).

## 6. The approximate index: usearch / HNSW

[usearch](https://github.com/unum-cloud/usearch) is a header-only C++11 library
(`usearch/index.hpp`). It is added with CMake `FetchContent`, exactly like
`s2geometry` and `spatialjoin` today (`CMakeLists.txt:197-207,369-371,463`); no
Conan recipe is needed. Relevant capabilities, all confirmed against its API:

- **High-level dense index** (`unum::usearch::index_dense_t`) with `add(key, ptr)`
  and `search(query_ptr, k)` returning keys + distances.
- **64-bit keys.** We use the entity's `VocabIndex` directly as the usearch key
  (it fits in 60 bits ≤ 64). Search results come back as `VocabIndex` values — no
  extra id↔row mapping is needed for the ANN path (unlike S2, which needs
  `shapeIndexToRow_`, `src/engine/SpatialJoinCachedIndex.h:22-90`).
- **Metrics**: cosine, L2², inner product, haversine, plus user metrics.
- **Quantisation**: f32/f16/bf16/i8/b1, matching §5.1.
- **Serialization**: `save(path)`, `load(path)`, and crucially **`view(path)` =
  memory-mapped, random-access, read-only** — so the HNSW graph is paged from
  disk like the flat store and survives restarts without rebuild.
- **Exact mode**: a brute-force search path (`exact=true`) that reuses the same
  metric — useful as a reference, though for the small-set optimisation we run
  our own brute force over the flat store (§8) to control the candidate set.

The HNSW index is built over the **same** vectors as the flat store and keyed by
`VocabIndex`. The flat store is retained even when HNSW exists, because it is
needed for (a) exact rerank / small-set search and (b) resolving a query entity's
vector.

## 7. The `VectorSearchJoin` operation

A new `class VectorSearchJoin : public Operation`
(`src/engine/VectorSearchJoin.{h,cpp}`) modelled directly on `SpatialJoin`. It is
a binary operation that joins a left (query) subtree with a right (search-space)
subtree.

### 7.1 Operation contract

The `Operation` virtuals to implement (signatures from `src/engine/Operation.h`):
`computeResult`, `getResultWidth`, `getCostEstimate`, `getSizeEstimateBeforeLimit`,
`getMultiplicity`, `resultSortedOn`, `getCacheKeyImpl`, `knownEmptyResult`,
`getChildren`, `getDescriptor`, `computeVariableToColumnMap`, `cloneImpl`. The
spatial join is the line-by-line template
(`src/engine/SpatialJoin.cpp:163-625`). Notes specific to vectors:

- **Output columns** (`computeVariableToColumnMap`, cf.
  `SpatialJoin.cpp:548-604`): all left columns, then the (payload-filtered) right
  columns, then an optional `?score` column appended last (a `Double`-typed
  `ValueId`). `getResultWidth` must agree (cf. `SpatialJoin.cpp:262-297`).
- **Cache key** (`getCacheKeyImpl`): both children's cache keys, the index name,
  `k`, metric, threshold, score-var flag, payload columns, and — because HNSW is
  *approximate* — **the chosen algorithm** (exact vs HNSW changes the result, so
  unlike the spatial join it must be in the key).
- **`resultSortedOn`**: `{}` (kNN output is not globally sorted; per-query-block
  ordering is not a global guarantee).

### 7.2 Two-phase construction (the planner trick)

The most important pattern to copy: a magic SERVICE produces a
**half-constructed binary operation** that advertises its not-yet-bound join
variable as an *undefined* output column, so the planner is forced to bind the
other side to it (`SpatialJoin::computeVariableToColumnMap`,
`SpatialJoin.cpp:548-561`; `addChild` returning a fresh instance,
`SpatialJoin.cpp:98-133`; `isConstructed`, `:136`). `VectorSearchJoin` reuses
this verbatim, with `vec:left`/`vec:right` (or `queryVector`/`vec:right`) as the
join handle.

The query vector is the one twist: when supplied **by entity** (`vec:left`), the
left side is a normal subtree and the join binds on the entity variable, exactly
like spatial join. When supplied **by literal** (`vec:queryVector`), there is no
left subtree — the operation has a single (right) child and emits a fixed query,
analogous to a spatial join whose right side comes from a cached index rather
than a subtree (`SpatialJoin.cpp:63-94`).

## 8. Search algorithms

Two algorithms, mirroring the spatial join's baseline-vs-S2 split
(`src/engine/SpatialJoinAlgorithms.h:108-112`). They live in a
`VectorSearchAlgorithms` helper constructed per `computeResult` from a
`PreparedVectorJoinParams` struct (cf. `SpatialJoin::prepareJoin`,
`SpatialJoin.cpp:463-513`).

1. **Exact** (`ExactAlgorithm`, the analogue of `BaselineAlgorithm`,
   `SpatialJoinAlgorithms.cpp:354-430`): materialise the right child to get a
   concrete candidate set of entity ids; for each left query vector, look up each
   candidate's vector in the flat store and compute the metric, keeping a bounded
   top-k heap. `O(|left| · |right| · D)`. Always available (needs only the flat
   store), always exact. This is the **only** path when no HNSW exists.

2. **HNSW** (`HnswAlgorithm`, the analogue of `S2geometryAlgorithm`,
   `SpatialJoinAlgorithms.cpp:613-676`): for each left query vector call
   `VectorIndex::searchHnsw(query, k, filter, maxDist)`. If the right side is the
   whole index, `filter` is empty. If the right side is a restricted set, pass a
   predicate filter so only matching entities are returned (usearch supports a
   per-candidate filter callback). Approximate; available only when the index has
   an HNSW file.

Both stream results into the output `IdTable` via an `addResultRow` helper
(cf. `addResultTableEntry`, `SpatialJoinAlgorithms.cpp:311-351`): left columns +
selected right columns + optional `Double` score.

## 9. Query optimisation — exact vs. approximate

This is the optimisation the brief calls out ("5 000 green statues → run exact").
The decision is driven by the **estimated cardinality of the right (search-space)
child**, which is already available to an operation at plan time via
`child->getSizeEstimate()` — exact for index-scan children
(`src/engine/IndexScan.cpp` `computeSizeEstimate`/`getExactSize`), and read by
the spatial join in precisely this way (`SpatialJoin.cpp:300-359`).

Policy for `vec:algorithm vec:auto` (the default):

- **No HNSW index for `N`** → always **Exact** (the only option).
- **Right side restricted and small** (`getSizeEstimate(right) ≤ θ`, a runtime
  parameter, default e.g. 50 000) → **Exact** over that set. It is both faster
  than filtered-HNSW for selective filters *and* gives exact recall. This is the
  "green statues" case.
- **Right side is the whole index, or large** → **HNSW** (optionally with a
  filter when restricted but still large).
- **Explicit `vec:exact` / `vec:hnsw`** overrides the policy.

Two clean places to apply it, both supported by the framework and both used by
the spatial join:

- **Plan time** (recommended): decide the algorithm in `addChild`/constructor
  from `child->getSizeEstimate()`, store it on the operation, and make
  `getCostEstimate` / `getSizeEstimateBeforeLimit` / `getCacheKeyImpl` reflect it
  so the planner scores the chosen variant and caches it correctly. SpatialJoin's
  estimates already branch on the algorithm this way (`SpatialJoin.cpp:300-370`).
- **Runtime** fallback: in `computeResult`, after materialising the right child,
  use the *actual* `idTable().numRows()` to confirm/override the choice.

Cost / size estimates (mirroring `SpatialJoin.cpp:300-411`):

- `getSizeEstimateBeforeLimit` = `|left| · k` (or `|left| · |right|` capped, with
  a damping constant like `SPATIAL_JOIN_MAX_DIST_SIZE_ESTIMATE`,
  `SpatialJoinConfig.h:112`).
- `getCostEstimate`: Exact ≈ `|left| · |right| · D`; HNSW ≈
  `|left| · (log|N| + k) · D` + children's costs.
- `getMultiplicity`: score column → 1; entity columns derived from the source
  child as in `SpatialJoin.cpp:373-411`.

A further future optimisation (noted, not required for v1): when the **left**
side is large and the **right** side is the whole index, batch all left queries
through one usearch call and exploit shared graph traversal; usearch is
thread-safe for concurrent search, so left rows can be fanned out across threads.

## 10. Build pipeline & lifecycle integration

The build/load lifecycle follows the **text index** precisely — it is QLever's
canonical "optional auxiliary index from separate input, extra files, optional
load" feature.

### 10.1 Building (`qlever index`)

- **CLI / config.** Add a `--vector-index` option to `IndexBuilderMain.cpp`
  taking a JSON spec (one entry per named index), exactly like
  `--materialized-views` (`src/index/IndexBuilderMain.cpp:290`,
  `parseMaterializedViewsJson` at `:149`). The spec carries, per index: name,
  input file(s), `dim`, `metric`, `scalar`, and HNSW build params
  (`connectivity`/M, `expansion_add`/efConstruction, etc.), plus
  `buildHnsw: true|false`.
- **Build pass.** After `index.createFromFiles(...)`
  (`src/libqlever/Qlever.cpp:88`), invoke a new `VectorIndexBuilder` — modelled on
  the text-index block (`Qlever.cpp:91-109`) and on `TextIndexBuilder` (a thin
  `IndexImpl` subclass, `src/index/TextIndexBuilder.h:12`). It:
  1. Reloads the on-disk vocabulary (the in-RAM vocab is freed during the main
     build — `TextIndexBuilder.cpp:49-55` shows the reload idiom) so it can
     resolve **URI → VocabIndex**.
  2. Streams `(URI, vector)` rows from the input file, resolving each URI to its
     `VocabIndex`; rows whose URI is not in the KG are skipped (with a warning
     count).
  3. Writes the sorted `.vec.<N>.keys` + `.vec.<N>.data` flat store via
     `MmapVector`.
  4. If `buildHnsw`, constructs a usearch `index_dense_t`, `add`s every row keyed
     by `VocabIndex`, and `save`s `.vec.<N>.hnsw`.
  5. Appends the index's entry to `configurationJson_["vector-indices"]` so
     `writeConfiguration()` persists it.

### 10.2 Loading (server / embedded start)

Add `IndexImpl::loadVectorIndicesFromDisk()` modelled on
`addTextFromOnDiskIndex()` (`src/index/IndexImpl.Text.cpp:23-57`): for each entry
in the manifest's `vector-indices`, `MmapVectorView`-open the `.keys`/`.data`
files and, if `hasHnsw`, `view()`-map the `.hnsw` file. Expose it on `Index`
(like `Index::addTextFromOnDiskIndex`, `src/index/Index.cpp:29`) and call it from
both `Server::initialize` (`src/engine/Server.cpp:93-96`) and the embedded
`Qlever` constructor (`src/libqlever/Qlever.cpp:41-44`). Loading is gated by the
manifest key (auto-detected) — no new server flag is strictly required, though a
`--no-vector-index` opt-out is cheap to add.

### 10.3 Updates

v1 treats vector indices as **read-only** auxiliary structures, exactly like the
text index (which is not updated by SPARQL `UPDATE`). Delta triples do not affect
vectors. Rebuilding/refreshing an index is an offline `qlever index` step. (A
later phase can support incremental `add`/`remove` since usearch supports both.)

## 11. Keeping it clean and maintainable

The design's guiding principle is **"a second instance of the spatial-join
pattern,"** which both proves the pattern generalises and keeps the new code
idiomatic. Concrete maintainability choices:

1. **Self-contained module.** All new engine code lives under
   `src/engine/` with a `VectorSearch*` prefix; all new storage/build code under
   `src/index/vectorIndex/`; the parser type under `src/parser/`. The
   dependency-light config struct (`VectorJoinConfig.h`) has no engine includes,
   so parser and engine share it without a cycle — exactly as
   `SpatialJoinConfig.h` does.

2. **Add new `SERVICE`s through a registry, not by editing core files.** Adding
   a magic service today means editing five core seams (the IRI table, the
   parser `if/else` dispatch, the closed `GraphPatternOperation` variant, the
   planner's `if constexpr` dispatch, and the hard-coded
   `checkSpatialJoin`/`createSpatialJoin` join hook). The
   [magic-service extensibility design](./magic-service-extensibility.md)
   proposes a one-time refactor (**Phase R**) that collapses the per-service
   hand-dispatch into (a) an IRI→factory **registry** on the parser side, (b) a
   virtual `MagicServiceQuery::plan()` on the planner side, and (c) a tiny
   `IncompleteBinaryJoin` interface (implemented by both `SpatialJoin` and
   `VectorSearchJoin`) that lets `createJoinCandidates` bind any incomplete
   binary magic-join generically. **After Phase R, adding a service — including
   the vector join — is 2–3 self-contained files plus one registration line,
   with zero edits to `GraphPatternOperation.h`, `SparqlQleverVisitor.cpp`, or
   `QueryPlanner.cpp`.** That document also assesses (and recommends against, for
   now) runtime `dlopen` plugin discovery, given QLever's all-static,
   no-stable-ABI, template-heavy build.

3. **No `indexFormatVersion` bump.** Vector files are additive; old indices keep
   working, new vector files version themselves (§5.2).

4. **Reuse, don't reinvent, the storage primitive.** `MmapVectorView<T>` already
   provides exactly the memory-mapped fixed-stride array we need
   (`src/util/MmapVector.h`); no bespoke mmap code.

5. **One optional, well-isolated dependency.** usearch is header-only and pulled
   via `FetchContent`; if it is ever undesired, the feature is compile-gated
   behind a CMake option (`QLEVER_WITH_VECTOR_INDEX`) and the exact-search path
   could even run without it.

### Optional later enhancement — vectors as first-class values

If we later want to *return* or *bind* a stored vector as a SPARQL value (e.g.
`SELECT ?vec`), add a `Datatype::VectorIndex` tag (4 enum slots are free,
`src/global/ValueId.h:32-50`) whose 60-bit payload is `(indexId, row)`. This is
purely additive on top of the storage above and is **not needed** for the kNN
join. It would let vectors participate in `BIND`, `FILTER`, export, etc., at the
cost of the usual `ValueId` plumbing (toString/visit/accessors).

## 12. Open decisions

These are flagged for confirmation before/within the implementation plan:

1. **Input format — DECIDED: Parquet.** v1 ingests Parquet
   `(uri: string, embedding: list<float>)` (the format ML pipelines emit). This
   pulls in **Apache Arrow** as a build dependency from P1 (added via
   `FetchContent`/Conan). The native on-disk store is still QLever's own
   memory-mapped flat format (§5.1); Parquet is only the *ingest* format. A
   lightweight `.npy`/`.npz` reader may be added later as a dependency-free
   alternative, but is not required.
2. **Query-vector mode — DECIDED: both, by-entity primary.** `vec:left ?entity`
   (look the vector up in the index) is the main path and the true binary join;
   `vec:queryVector` (inline external embedding) is also supported (§4.1, §7.2).
3. **SPARQL surface — DECIDED: magic `SERVICE`** (matches spatial/text/path
   search), added via the registry from
   [magic-service-extensibility.md](./magic-service-extensibility.md).
4. **Default similarity score semantics** (open) — return raw metric distance, or
   a normalised `[0,1]` similarity for `?score`? Recommendation: bind raw metric
   distance as `?score` and document per-metric meaning (cosine distance,
   L2², …), matching the spatial join's raw-distance approach.
```
