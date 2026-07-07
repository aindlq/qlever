# Vector search in QLever — SPARQL usage & how it differs from PR #3043

Two parts: (1) how an end user drives it from SPARQL, use case by use case;
(2) how the storage/query mechanism differs from the in-house embeddings PR
[ad-freiburg/qlever#3043](https://github.com/ad-freiburg/qlever/pull/3043), and
why. For building indices see [`indexing.md`](indexing.md); for the rationale
see [`index-payload-design.md`](index-payload-design.md).

## Part 1 — Using it from SPARQL

Every query uses these prefixes:

```sparql
PREFIX vec:  <https://qlever.cs.uni-freiburg.de/vectorSearch/>
PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>
```

An **index** is referenced by its IRI `vidx:<name>` — the *same* IRI you
introspect and the one you search. The three functions:

| function | meaning |
|---|---|
| `vec:distance(vidx:X, S1, S2)` | uniform **smaller-is-closer** distance between two *sources*; each source is an entity (→ its stored vector) or a query vector |
| `vec:embed(vidx:X, input)` | embed text / an image IRI via index X's configured endpoint → a query vector |
| `vec:vector(vidx:X, ?e)` | entity `?e`'s stored vector *from index X* as a query vector (for cross-index) |

The metric (cosine / dot / l2) is a property of the index, so there is **one**
distance function — you never pass a metric, and every ranking is `ORDER BY ?d ASC`.

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

### Use case 4 — a precomputed query vector (inline)
```sparql
BIND(vec:distance(vidx:emb, ?e, "0.12,-0.03, … ,0.44") AS ?d)
```
A comma-separated float string is parsed as the query vector (dimension checked).

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
(**model + precision + dimension**) → matches compute, a mismatch is UNDEF, not a
garbage number. (Simplest alternative: if both are the same model, put them in
**one** index and `vec:distance(vidx:visual, ?a, ?p)`.)

### Use case 6 — whole-index top-k, exact (no index, no SERVICE)
```sparql
SELECT ?doc ?d WHERE {
  ?doc <hasEmbedding> ?anything .                     # any pattern that enumerates the members
  BIND(vec:distance(vidx:emb, ?doc, vec:embed(vidx:emb, "a red bicycle")) AS ?d)
  FILTER(BOUND(?d))
} ORDER BY ?d LIMIT 10
```
Search is just `ORDER BY distance LIMIT k`. Fine to a few million vectors.

### Use case 7 — whole-index top-k, accelerated (HNSW `SERVICE`)
For huge unfiltered top-k, the opt-in HNSW graph via the `SERVICE` block (exact
never needs it; HNSW is always explicit). See `indexing.md` for the `hnsw:true`
build key and the SERVICE shape.

### Use case 8 — introspect an index (it's a real RDF resource)
```sparql
SELECT ?model ?dim ?metric ?count WHERE {
  vidx:emb  vec:model ?model ; vec:dimension ?dim ; vec:metric ?metric ; vec:count ?count .
}
```

### Getting vectors in (one line)
No RDF for the vectors — a `.npy` matrix (fp32 or ml_dtypes bf16) + a bare-IRI
`.iris` list, fed to `qlever-index --service-index '{"vectorSearch":[{…}]}'`
(keys: `name, npy, iris, metric, scalar, hnsw, embeddingUrl, embeddingModel`).
For normalized embeddings (SigLIP2, Qwen3): `metric:"dot"`, `scalar:"bf16"`.

---

## Part 2 — How our mechanism differs from PR #3043, and why

Both projects independently reached the *same core insight* — a vector is a
first-class thing decoded to a compact sidecar, not re-parsed per query, and the
embedding "type" is a queryable RDF resource. Where we diverge is **what the
vector *is*** and **how it's laid out**, and that drives everything else.

### Storage

| | **PR #3043** | **Ours (index-payload)** |
|---|---|---|
| A vector is… | a first-class **RDF literal** `"[…]"^^emb:fp32Vector` | **index payload** — bytes in a `VectorIndex`, *not* an RDF term |
| Where it lives | the vocabulary (`EmbeddingVocabulary`) + a `.embvec` sidecar | an **entity-keyed, per-space, contiguous, fixed-stride** array (+ optional HNSW) |
| Layout | one blob in **vocabulary order**, offset-table addressed | a packed SIMD-aligned matrix per `(model,dim,precision)` space |
| Per-embedding RDF | reified: `<e> emb:hasEmbedding _:b . _:b emb:asFp32Vector … ; emb:type …` | **none** — vectors never enter the vocab/permutations/triples |
| Precision | fp32 | f32 / f16 / **bf16** / i8 |

**Why not RDF literals.** We pressure-tested exactly #3043's choice and dropped
it, for three concrete reasons:
1. **No real use case** for `SELECT`/`CONSTRUCT` of a raw vector — entity-to-entity
   similarity passes the *entity* (the function looks the vector up), bulk export
   wants a binary dump not float-text, arithmetic is app-layer.
2. **The literal wastes storage and negates precision.** The decimal/JSON text is
   full-precision regardless of storage precision, so an `i8` vector (512 bytes)
   carries a ~4 KB text twin — ~8× the payload, defeating quantisation.
3. **Per-embedding reification doesn't scale.** #3043's blank-node modeling is
   ~3 triples + a blank node per embedding, materialised across all six
   permutations — ≈ 1.8 B permutation entries at 100 M vectors, versus **zero**
   for us. The type is a property of the *space*, not the instance.

**Why a contiguous per-space array.** #3043's `.embvec` is one vocabulary-ordered
blob addressed by an offset table — reading the vectors for a query is a
**random** positioned read per vector, models interleaved. Ours is a contiguous,
fixed-stride, SIMD-aligned matrix per space, so a brute-force scan streams
sequentially through NumKong's per-precision SIMD kernels (and it *is* the HNSW
graph's backing). That contiguous array is the real "index" — even without ANN.

### Query

| | **PR #3043** | **Ours** |
|---|---|---|
| Distance | one `embf:distance(a, b, type)`; metric from the `type` resource | one `vec:distance(<idx>, S1, S2)`; metric from the index |
| Query vector | a materialised RDF literal you pass in | a *transient* value from `vec:embed`/`vec:vector`/inline — never a stored term |
| Query-time embedding | **none** — precompute externally (a Python script) | **`vec:embed`** — text/image via a server endpoint (http(s) **or** `unix:` socket) |
| ANN | **none** — brute-force only | optional **HNSW** via `SERVICE` (brute-force is the default) |
| Comparability | pass the `type` IRI; **hard error** on mismatch | model+precision carried in the query-side **typed literal**, validated (+ dim) → **UNDEF** on mismatch |
| Introspection | `emb:type` resource with metric/dim/precision triples | `vidx:X` resource with `vec:model/dimension/metric/count` triples |

**Why these differ.**
- **Query-time embedding.** #3043 can only rank against a vector you computed
  offline; you can't "search by text/image" from SPARQL. We embed at query time
  (`vec:embed`), so a plain text or image query works — and because the endpoint
  is bound to the index, the query lands in the same space that built it.
- **HNSW.** #3043 is exact-only. We keep exact brute-force as the default (it's
  just `ORDER BY distance LIMIT`) and add HNSW as an explicit opt-in for
  unfiltered top-k over an index too large to scan. Search is never a magic
  operator; it's the same SPARQL, sub-linear only when you ask.
- **Comparability via a transient typed literal.** #3043 makes you name the
  `type` and hard-errors on mismatch. We keep the vector out of RDF, but let the
  *query-side* value (from `vec:embed`/`vec:vector`) be a transient typed literal
  `^^<…/vec/model/precision>`; `vec:distance` validates it against the index
  (model + precision + dim) and returns UNDEF on mismatch — which is what makes
  **cross-index** distance safe, and is SPARQL-idiomatic (a bad row drops, the
  query survives).
- **Introspection — where we converged.** Both make the space a queryable RDF
  resource. We adopted that idea, but attach the metadata **once per index**
  (auto-materialised at load) rather than per embedding.

### One-line summary
#3043 stores vectors *as RDF you can SELECT* and searches them pairwise, exact,
from precomputed literals. We store vectors *as searchable index payload* — a
compact per-space SIMD array you never SELECT — reached by one distance function,
embedded at query time, optionally HNSW-accelerated, with the index as the only
RDF surface. Same native-first instinct; opposite answer to "is a vector a term
or a payload," and that choice is what buys the storage, the scan speed, the
query-time embedding, and the ANN path.
