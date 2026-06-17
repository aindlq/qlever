# Vector index integration — implementation plan

> Companion to [`architecture.md`](./architecture.md). This document is the
> phased, file-by-file plan. It is sequenced so that **every phase is
> independently useful, mergeable, and testable** — you get exact similarity
> search working before any ANN/HNSW or planner complexity exists.

## Guiding sequencing principle

Build outward from a working core, and isolate the extension mechanism *first*
so the vector feature never edits core parser/planner files:

```
R  magic-service registry refactor (behaviour-preserving)   <- prerequisite
P0 dependency (usearch + Arrow) + skeleton
P1 native storage + exact brute-force search                <- usable end-to-end, no HNSW
P2 HNSW build + load + approximate algorithm
P3 cardinality-aware exact/approx optimisation
P4 ergonomics: quantisation, multi-index polish, optional .npy/.npz input
P5 (optional) vectors as first-class values; incremental updates
```

**Phase R** (see [`magic-service-extensibility.md`](./magic-service-extensibility.md))
is the user's explicit requirement — "don't mangle core files when adding
SERVICEs." It is a standalone, behaviour-preserving refactor that migrates the
existing magic services (spatial/text/path/…) onto a registry + virtual
`plan()` + `IncompleteBinaryJoin` interface, guarded by the existing test suite.
After R, the vector feature is added through the registry with **zero edits** to
`GraphPatternOperation.h`, `SparqlQleverVisitor.cpp`, or `QueryPlanner.cpp`.

After **P1** a user can already create a vector index and run exact kNN joins —
exactly the "small database, no index needed" requirement. HNSW (P2) and the
optimiser (P3) are pure scale add-ons layered on a working feature.

> Ordering note: this is ordering **(A)** from the extensibility doc
> (refactor-first). If a working vector index is wanted sooner, ordering **(B)**
> defers R: implement P1–P3 by editing the five seams as `SpatialJoin` does
> today, then extract the registry once two binary magic joins exist. (A) is
> recommended because only it keeps the vector feature itself free of core-file
> edits.

---

## Phase R — Magic-service registry refactor (prerequisite)

**Goal:** new magic `SERVICE`s are added via registration, not by editing core
files. Behaviour-preserving — no new feature. Full design in
[`magic-service-extensibility.md`](./magic-service-extensibility.md) §2–§3.

- Add `MagicServiceQuery::plan(GraphPatternPlanner&)` virtual; move each existing
  service's planner branch (`QueryPlanner.cpp:3135-3147`) into its
  `XQuery::plan()`.
- Introduce `parsedQuery::MagicService { unique_ptr<MagicServiceQuery> }`;
  collapse the six magic-service alternatives in the `GraphPatternOperation`
  variant (`GraphPatternOperation.h:232-236`) into it; update the single
  exhaustive visitor (`graphPatternOperationVisitor`) and the few targeted
  `std::get<XQuery>` sites.
- Add `MagicServiceRegistry` (IRI/prefix → factory) and a
  `registerBuiltinMagicServices()`; replace the IRI `if/else`
  (`SparqlQleverVisitor.cpp:1285-1304`) with a registry lookup.
- Extract `IncompleteBinaryJoin` (`isConstructed`, `addChild`, join vars) from
  `SpatialJoin`; make `createJoinCandidates`/`createSpatialJoin`
  (`QueryPlanner.cpp:2263-2401`) operate on the interface.

**Deliverable:** identical behaviour; existing spatial/text/path/external-values/
materialized-view/named-cache services run through the registry.
**Tests:** the entire existing test suite is the safety net; add a tiny
registry-lookup unit test. Each sub-step lands independently green.

---

## P0 — Dependencies and module skeleton

**Goal:** usearch + Arrow compile and link; empty modules exist.

- `CMakeLists.txt`: add `FetchContent_Declare(usearch GIT_REPOSITORY
  https://github.com/unum-cloud/usearch ...)` and add it to
  `FetchContent_MakeAvailable(...)` (`CMakeLists.txt:197-207,463`). Guard the
  whole feature behind `option(QLEVER_WITH_VECTOR_INDEX "..." ON)`.
- **Apache Arrow / Parquet** dependency (for ingest, per the decided input
  format): add via Conan (`conanfile.txt`, e.g. `arrow/…` with `parquet=True`)
  or `FetchContent`. This is the one heavy dependency; keep it confined to the
  build-time `VectorInputReader` translation unit so the query path doesn't link
  it at runtime.
- Create directories/targets: `src/index/vectorIndex/` (add to
  `src/index/CMakeLists.txt`), and register new engine/parser files in their
  `CMakeLists.txt`.
- Add `VectorJoinConfig.h` (dependency-light enums + `VectorJoinConfiguration`
  struct), mirroring `src/engine/SpatialJoinConfig.h`.

**Deliverable:** clean build with the option on/off. No behaviour change.
**Test:** CI builds in both modes; Arrow links only into the builder.

---

## P1 — Native storage + exact search (the MVP)

**Goal:** `qlever index` can build a flat vector store from an input file; a
`SERVICE vec:` query runs an **exact** kNN join. No HNSW yet.

### Storage (`src/index/vectorIndex/`)

- `VectorIndexFormat.h` — file-suffix constants (`.vec.<N>.{meta,keys,data}`),
  `VECTOR_INDEX_VERSION`, the `.meta` JSON schema, `VectorMetric`/`ScalarType`
  enums.
- `VectorIndex.{h,cpp}` — the runtime accessor (see architecture §5.3):
  `MmapVectorView`-backed `keys_`/`data_`, `getVector(VocabIndex)`,
  `searchExact(query, k, candidates, maxDist)`. (HNSW members stubbed.)
- `VectorIndexBuilder.{h,cpp}` — writes the sorted `.keys`/`.data` via
  `MmapVector` from a stream of `(VocabIndex, vector)` rows; writes `.meta`.
- `VectorInputReader.{h,cpp}` — reads **Parquet** `(uri: string, embedding:
  list<float>)` via Apache Arrow, streaming row groups (never loads the whole
  file). Yields `(string uri, span<float> vec)`. Validates dimension consistency.
  Arrow is linked only here.

### Build wiring

- `IndexBuilderConfig` (`src/libqlever/Qlever.h:73`): add
  `std::vector<VectorIndexSpec> vectorIndices_` (or a parsed JSON blob).
- `IndexBuilderMain.cpp`: add `--vector-index` option + a `parseVectorIndexJson`
  (model on `parseMaterializedViewsJson`, `:149,290`).
- `Qlever::buildIndex` (`src/libqlever/Qlever.cpp:88`): after
  `index.createFromFiles(...)`, run `VectorIndexBuilder` per spec. It reloads the
  vocab (idiom from `TextIndexBuilder.cpp:49-55`) to resolve URI→VocabIndex,
  builds the store, and appends the manifest entry to
  `configurationJson_["vector-indices"]`.

### Load wiring

- `IndexImpl` (`src/index/IndexImpl.h:126-129`): add
  `HashMap<string, VectorIndex> vectorIndices_` + accessor.
- `IndexImpl::loadVectorIndicesFromDisk()` (model on
  `addTextFromOnDiskIndex`, `IndexImpl.Text.cpp:23`): read manifest key, mmap
  each store.
- Expose on `Index` (`Index.{h,cpp}`); call from `Server::initialize`
  (`Server.cpp:93-96`) and `Qlever` ctor (`Qlever.cpp:41-44`).

### Parser (via the Phase R registry — no core edits)

- `VectorSearchQuery.{h,cpp}` — `: MagicServiceQuery`; `addParameter`,
  `validate`, `name`, `toVectorJoinConfiguration()`, and the `plan()` virtual
  (model on `SpatialQuery.{h,cpp}`). `plan()` plans the nested pattern and
  constructs the `VectorSearchJoin` (mirrors today's `visitSpatialSearch`).
- Register it: add `VECTOR_SEARCH_IRI` and a line in
  `registerBuiltinMagicServices()`. **No edits to `GraphPatternOperation.h`,
  `SparqlQleverVisitor.cpp`, or `QueryPlanner.cpp`** (that's the point of R).

### Operation (exact only)

- `VectorSearchJoin.{h,cpp}` — `: Operation` and implements
  `IncompleteBinaryJoin` (from Phase R); two-phase construction copied from
  `SpatialJoin` (`addChild`, `isConstructed`, undefined-column map). All
  `Operation` virtuals (architecture §7.1). `computeResult` calls only the exact
  algorithm. Because it implements `IncompleteBinaryJoin`, the generic
  `createJoinCandidates` binds its children automatically — no planner edit.
- `VectorSearchAlgorithms.{h,cpp}` — `ExactAlgorithm` (model on
  `BaselineAlgorithm`, `SpatialJoinAlgorithms.cpp:354-430`) + `addResultRow`.

**Deliverable:** end-to-end exact vector kNN join, single or multiple named
indices, `.npy`/`.npz` input.

**Tests:**
- Unit: `VectorIndex` round-trip (build → mmap → `getVector`/`searchExact`);
  builder URI-resolution incl. unknown-URI skip; input reader.
- Unit: `VectorSearchJoin` column map / width / cache key / estimates;
  `ExactAlgorithm` correctness vs. a NumPy reference (model on
  `test/engine/SpatialJoinAlgorithmsTest.cpp`).
- E2E: a tiny KG + tiny `clip` index, queries A/B/C from architecture §4.2.

---

## P2 — HNSW (approximate) index

**Goal:** optionally build and use a usearch HNSW index for scale.

- `VectorIndexBuilder`: when `buildHnsw`, construct `index_dense_t`, `add(key=
  VocabIndex, ptr)` per row, `save()` to `.vec.<N>.hnsw`; set `hasHnsw:true` in
  the manifest.
- `VectorIndex`: `view()`-map the `.hnsw` file on load; implement `searchHnsw`
  (with optional predicate filter for restricted right sides).
- `VectorSearchAlgorithms`: add `HnswAlgorithm` (model on `S2geometryAlgorithm`,
  `SpatialJoinAlgorithms.cpp:613-676`).
- `VectorSearchJoin::computeResult`: dispatch on the configured algorithm;
  `getCacheKeyImpl` must include the algorithm (HNSW is approximate).
- Config: honour explicit `vec:exact` / `vec:hnsw`; `vec:auto` still defaults to
  exact in P2 (auto-switching is P3).

**Deliverable:** approximate search available; `vec:hnsw` works.
**Tests:** recall@k of HNSW vs. exact on a synthetic dataset (assert ≥ threshold);
`view()` mmap load; quantised (f16) store parity within tolerance.

---

## P3 — Cardinality-aware optimisation

**Goal:** `vec:auto` picks exact vs. HNSW from estimated cardinality. (The
planner generalisation already happened in Phase R via `IncompleteBinaryJoin`.)

- **Auto policy** (architecture §9): in `VectorSearchJoin::addChild`/ctor read
  `rightChild_->getSizeEstimate()`; choose Exact if no HNSW or if the right side
  is restricted and `≤ θ`; else HNSW. Store the choice; reflect it in
  `getCostEstimate`/`getSizeEstimateBeforeLimit`/`getCacheKeyImpl`.
- Runtime guard in `computeResult`: re-check against the materialised right size.
- Add a `RuntimeParameters` entry for `θ`
  (`src/global/RuntimeParameters.{h,cpp}`).
- Estimates: implement the cost/size/multiplicity formulas (architecture §9,
  mirroring `SpatialJoin.cpp:300-411`).

**Deliverable:** the "5 000 green statues → exact" optimisation.
**Tests:** planner picks Exact for a selective right side and HNSW for an
unrestricted one (assert via the chosen algorithm in `RuntimeInformation` /
cache key).

---

## P4 — Ergonomics

- **Quantisation** end-to-end (f16/i8/b1): builder + store + HNSW + exact metric
  paths honour `scalar`.
- **Multi-index polish:** clear errors for unknown index / dim mismatch / metric
  mismatch; `qlever index` summary stats per vector index; docs + example
  dataset.
- **Filtered HNSW tuning:** choose filtered-HNSW vs. exact for medium-selectivity
  right sides based on measured crossover.
- **Optional `.npy`/`.npz` reader:** a dependency-free ingest alternative to
  Parquet (`.npy` matrix + row-aligned IRI list), for pipelines that prefer not
  to emit Parquet.

---

## P5 — Optional, future

- **Vectors as first-class values:** `Datatype::VectorIndex` tag
  (`ValueId.h:32-50`) for `BIND`/`FILTER`/export of stored vectors
  (architecture §11). Purely additive.
- **Incremental updates:** usearch supports `add`/`remove`; wire vector
  add/remove to entity lifecycle if/when needed. v1 is read-only like the text
  index.
- **Batched / multi-threaded left-side search**, GPU build, product
  quantisation, etc.

---

## Cross-cutting concerns

- **Versioning:** never bump `indexFormatVersion`
  (`src/index/IndexFormatVersion.h:42`) for vector files; use per-index
  `VECTOR_INDEX_VERSION` + manifest presence (architecture §5.2).
- **Cache correctness:** the algorithm (exact/HNSW) and all result-affecting
  params must be in `getCacheKeyImpl` (HNSW is approximate). Wrong cache keys
  silently corrupt results — this is the highest-risk correctness item.
- **Definedness:** `?score` is `AlwaysDefined` (a real `Double`); result-width and
  the var-to-column map must match what `computeResult` emits or
  expensive-checks builds assert (`Operation.cpp:168-179`).
- **Memory:** mmap everything (`MmapVectorView` + usearch `view()`); never load a
  full index into RAM. Use `AccessPattern::Random`.
- **Embedded library parity:** wire build + load in both the server
  (`ServerMain`/`Server`) and the embedded `libqlever` paths
  (`Qlever.cpp`), as the text index does.

## Effort sketch (rough)

| Phase | Scope | Relative size |
| --- | --- | --- |
| R | magic-service registry refactor (behaviour-preserving) | M |
| P0 | deps (usearch + Arrow) + skeleton | S |
| P1 | storage + exact + Parquet ingest + full plumbing | L (the bulk) |
| P2 | HNSW build/load/search | M |
| P3 | cardinality-aware optimiser | S–M |
| P4 | quantisation, polish, optional .npy | M |
| P5 | first-class values / updates | M (optional) |

The largest single chunk is **P1**, because it threads the whole
parser→planner→operation→index→build→load path for the first time; P2–P3 then
slot algorithms and policy into established seams. **Phase R** is independent of
the vector work and can be reviewed/merged on its own.

## Key risks & mitigations

| Risk | Mitigation |
| --- | --- |
| Apache Arrow bloats the build / runtime | Confine Arrow to the build-time `VectorInputReader` translation unit; the query path never links it. usearch is header-only. An optional `.npy`/`.npz` reader (P4) avoids Arrow entirely if ever desired. |
| Approximate results cached as if exact | Algorithm in cache key; auto-switch reflected in key (P3). |
| HNSW with selective filter is slow/low-recall | Auto policy routes small/selective right sides to exact (P3). |
| Vocab freed mid-build | Build pass reloads vocab from disk (TextIndexBuilder idiom). |
| Adding services keeps mangling core files | Phase R registry + `IncompleteBinaryJoin`: new services are self-contained files + one registration line. |
| Registry refactor breaks existing services | Phase R is behaviour-preserving and guarded by the full existing test suite; each sub-step lands green independently. |
| Large vectors blow RAM | mmap-only via `MmapVectorView` and usearch `view()`. |
```
