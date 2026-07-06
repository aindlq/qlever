# Vector search in QLever: the index-payload design

Status: **agreed target** (2026-07). Branch: `vector-index-payload` (off
`vector-store-redesign`). This supersedes the native-vocab-literal exploration on
branch `vector-native-vocab`, which is kept only as a record of a rejected path.

## The decision in one line

**A vector is index payload you *search*, not RDF you *store*.** Embeddings are not
first-class RDF terms; they live in a named, entity-keyed index and are reached only
through a distance function and (optionally) an HNSW `SERVICE`. The index itself is a
queryable RDF resource carrying thin metadata triples.

## Why (the reasoning that got us here)

We pressure-tested making vectors first-class RDF literals (the `vector-native-vocab`
branch) and concluded it buys nothing anyone actually uses, at real cost:

- **No use case for `SELECT`/`CONSTRUCT` of a raw vector.**
  - *Entity-to-entity similarity* ("images like this one") is cleaner as
    `vec:distance(vidx:images, ?y, <X>)` — pass the entity, the function looks up both
    vectors internally. Materialising X's vector would be a step backwards.
  - *Bulk export for downstream ML* is real, but SPARQL `SELECT` is the wrong tool
    (512 floats × N as text); it wants a binary export, which argues *against* literals.
  - *Vector arithmetic* is niche and app-layer; QLever is not a vector-algebra engine.
  - *Debugging* looks at distances, not raw floats.
- **Storing the literal wastes storage and negates precision.** The decimal/JSON text
  form is full-precision regardless of storage precision, so an `i8` vector (512 bytes)
  carries a ~4 KB text twin — ~8× the payload — defeating quantisation end to end.
- **The literal layer needs `VectorVocabulary` + split-vocab + marker bits + a
  value-getter**, all of which exist *only* to make a vector `SELECT`-able.

So we drop vectors-as-terms. The one case that genuinely wants a vector value —
passing a *precomputed* query vector inline — is a string argument to `vec:distance`,
parsed internally, not a first-class term.

## The model

- An **index** (`idx:images`) is a named store: an **entity-keyed, per-space,
  contiguous, bytes-only** array of vectors (one `(model, dim, precision, metric)`
  space per index), plus an optional HNSW graph over it. This is the existing
  `VectorIndex`/`VectorIndexBuilder` from `vector-store-redesign`.
- The index is also a **queryable RDF resource**: a handful of auto-materialised
  metadata triples (below), once per index — never per embedding.
- Vectors never enter the vocabulary, the permutations, or any triple.

## Input: the `.npy` bundle (no Parquet, no decimal Turtle)

Per index/space, three tiny artefacts (already ingested by `buildFromNpy`):

```
images.npy        # the "one big array": N × D raw floats (C-order); descr gives precision, shape gives dim/count
images.uris.txt   # N lines, IRI per line, row i ↔ matrix row i
images.meta.json  # {"name":"images","model":"clip","metric":"cosine","precision":"fp16","hnsw":true}
```

`.npy` is zero-dependency (a 6-byte magic + one-line ASCII header + raw bytes; ~30 lines
to parse), native to any ML pipeline (`np.save`), and its C-order matrix *is* our
per-space array — so for real (×64) dimensions ingest is near-zero-copy (an odd
dimension re-lays each row into the padded store). Parquet (Arrow) is dropped; a base64
vector literal in Turtle stays available for the "just give me a `.ttl`" path but is
not the bulk route.

## Alignment & residency (SIMD scans)

The flat store is a contiguous fixed-stride matrix, `mmap`'d and scanned by NumKong's
per-precision SIMD kernels. Both things that keep those scans fast are free in practice:

- **Alignment.** `mmap` returns a page-aligned base, and real embedding dimensions are
  multiples of 64 (384, 512, 768, 1024, 1536, …), so the natural row stride
  (`dim × precision-width`) is already a multiple of 64 at every precision — every row
  starts 64-byte aligned and no SIMD load straddles a cache line, for free. NumKong
  kernels use unaligned loads + masked tails, so *any* dimension is correct regardless;
  alignment is just the bonus real data hands us. (Format v5 carries an explicit per-row
  stride that can pad an odd dimension as a safety net — real vectors never need it.)
- **Residency.** Because the store is `mmap`'d, a cold whole-index scan page-faults. For
  a query-hot index that fits in RAM, pin it (`mlock`, or a huge-page-backed aligned RAM
  copy — `Residency` in `VectorIndex.h`); leave it on the OS page cache otherwise. A
  per-index build-spec knob, off by default.

The builder reads `(uri, row)`, resolves the IRI to an entity id, and stores the vector
in the index array keyed by that entity — **no triples are created for the vectors.**

## Query surface

### 1. `vec:distance(<…/index/NAME>, S1, S2)` — the composable primitive

Returns a **uniform smaller-is-closer distance** (cosine → `1−cos`, dot → negated,
l2 → distance) between two vector *sources*. The first argument is the index **IRI**
(the same resource the metadata triples live on, so the thing you search is the thing
you can introspect). Each source is an expression yielding, per row, either

- an **entity** → its stored vector is looked up, or
- a **precomputed vector** → a comma-separated float string, parsed internally
  (`vec:embed(<…/index/NAME>, input)` produces exactly this form via the index's
  configured endpoint/model — a text literal embeds as text, an IRI as an image URL).

Anything else — including an entity not in the index — makes the row **UNDEF** (so
filtering on the index is implicit). A constant source is resolved once per query, not
per row. The metric is a property of the index (from its metadata), so there is **one**
distance function; you never pass a metric. Every search is therefore
`ORDER BY ?d ASC LIMIT k`.

**Filtered / composed** — entities come from your own patterns:
```sparql
SELECT ?art ?d WHERE {
  ?art a <Painting> .
  BIND(vec:distance(vidx:images, ?art, vec:embed(vidx:images, "a red bicycle")) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 10
```
Only the filtered entities are looked up; non-members drop via `FILTER(BOUND)`. Exact,
no index scan, no HNSW.

**Whole-index brute-force, no SERVICE** — the entities are enumerated by whatever real
pattern the embeddings were derived from (their type/predicate):
```sparql
SELECT ?art ?d WHERE {
  ?art <hasImage> ?img .                                     # the natural enumerator
  BIND(vec:distance(vidx:images, ?art, vec:embed(vidx:images, "a red bicycle")) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 10
```
Exact, brute-force over the whole set. No `qv:member` virtual predicate is needed: the
index array is entity-keyed, so distance looks up by entity and UNDEFs non-members.

### 2. `SERVICE` — HNSW only

The `SERVICE` means exactly one thing: *use the HNSW graph* for sub-linear whole-index
top-k. An expression cannot invoke an ANN top-k, and we deliberately do **not** let the
planner silently rewrite `ORDER BY vec:distance … LIMIT` into a graph lookup. So exact
never needs the `SERVICE`; HNSW is always an explicit opt-in.

### 3. `idx:` metadata triples — auto-materialised, queryable, thin

Once per index (not per embedding), the build emits real triples on the index resource,
so the space is introspectable in plain SPARQL:
```sparql
SELECT ?model ?dim ?metric ?count WHERE {
  idx:images  qv:model ?model ; qv:dimension ?dim ; qv:metric ?metric ; qv:count ?count .
}
```
These are ordinary RDF (kept out of the internal `builtin-functions/` namespace so they
survive index building), harvested into an in-memory registry at load for O(1) lookup —
the triples are the source of truth, the registry is a cache (the model #3043 uses, but
per *index* rather than per *embedding*).

## What we deliberately do NOT have

- **Vector RDF literals / a vector vocabulary.** (No `SELECT` use case; storage waste.)
- **Per-embedding triples / blank-node reification** (#3043's `emb:hasEmbedding` → `_:b →
  emb:asFp32Vector` → `emb:type`). At 100M vectors that is ~300M triples + 100M blank
  nodes across six permutations; the type is a property of the *space*, not the instance.
- **Virtual predicates** (`qv:embedding`, `qv:hasMember`, `qv:member`). A generic
  entity→vector predicate materialises vectors as join rows and, unfiltered over two
  10M indices, produces 20M rows to filter to 10M. Distance-as-a-function avoids
  materialising vectors at all; the natural entity pattern enumerates for whole-index.

## Reuse vs. new

**Reused verbatim from `vector-store-redesign`:** `VectorIndex`/`VectorIndexBuilder`
(entity-keyed flat store + HNSW), `buildFromNpy` + `VectorInputReader`, the NumKong SIMD
per-precision kernels, `VectorDistanceExpression` (`vec:distance`, since generalized to
two sources + the separate `vec:embed`), the `SERVICE` (`VectorSearch`,
`vec:algorithm exact|hnsw`), the embed client.

**New (this branch):**
1. **Auto-materialised `idx:` metadata triples** at build time (model / dimension /
   precision / metric / count on the index resource) + the load-time registry over them.
2. **The `.npy` bundle metadata** (`.meta.json`: model / metric / precision / name) so a
   bundle fully describes its space.
3. **End-to-end tests**: `.npy` bundle → build → `vec:distance` ranking →
   `SELECT` the `idx:` metadata.
4. Documentation (this file).

## Open / deferred

- Binary export op (entity-set → matrix) for the bulk "get vectors out" case.
- A query-less `SERVICE` mode to enumerate an index's member entities, *if* a real need
  appears (YAGNI: `qv:count` covers the common form).
- Uniform-distance polish for `vec:distance` if any metric still returns similarity.
- Expose `residency` (mlock / aligned huge-page copy) as a per-index build-spec key so
  hot indices can be pinned; the mechanism already exists (`VectorIndex.h`).
