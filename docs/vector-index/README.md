# Built-in vector similarity search for QLever

This directory documents a design for adding **native vector indices** to QLever
so that approximate/exact nearest-neighbour search can be **joined with ordinary
SPARQL graph data** in a single query — e.g. *"for each green statue, find the 10
most visually similar artworks"* against a CLIP image index, while a separate
Qwen3 index serves textual-metadata similarity.

It is the built-in equivalent of the external pattern described in the
[Pharos technical white paper](https://github.com/ArtResearch/artresearch.net/blob/dev/whitepaper/Pharos_Technical_White_Paper.md),
implemented inside QLever using [usearch](https://github.com/unum-cloud/usearch)
(HNSW) for the approximate index and a memory-mapped flat store for native vector
storage and exact search.

> **New to QLever's internals?** Start with the
> [**QLever from the Ground Up walkthrough**](../qlever-walkthrough.md) — an
> educational tour of how the index and engine work for someone who has never seen
> the codebase, ending with how this vector feature plugs in as a case study.

## Read in this order

1. **[`architecture.md`](./architecture.md)** — how QLever is extended cleanly:
   the data model and SPARQL surface, native vector storage, the usearch/HNSW
   index, the `VectorSearchJoin` operation, the exact-vs-approximate optimisation,
   build/load lifecycle, and the maintainability choices. Grounded throughout in
   the existing **spatial join**, **geo vocabulary**, and **text index** code,
   with `file:line` references.
2. **[`magic-service-extensibility.md`](./magic-service-extensibility.md)** —
   investigation of making magic `SERVICE`s **isolated and extensible** so adding
   one needs no edits to core parser/planner files. Proposes a one-time
   **registry refactor (Phase R)**; the vector join is then added through it. Also
   assesses runtime `dlopen` plugin discovery (and recommends against it for now).
3. **[`integration-plan.md`](./integration-plan.md)** — the phased,
   file-by-file implementation plan (Phase R, then P0–P5), where each phase is
   independently mergeable and testable. Exact search works end-to-end after
   **P1**, before any HNSW or planner complexity.

## Confirmed decisions

- **Ingest format:** Parquet `(uri: string, embedding: list<float>)` (Apache
  Arrow, build-time only). Native storage stays QLever's own memory-mapped flat
  format.
- **Query mode:** both *by-entity* (`vec:left ?x`, the primary path) and *inline
  vector* (`vec:queryVector`).
- **Extensibility:** magic `SERVICE` via a registry (Phase R); compile-time
  isolation, no runtime plugin loading.

## One-paragraph summary

A vector index is an **optional auxiliary index attached to existing entities**
(URI → dense vector), built in a separate `qlever index` pass from `(URI, vector)`
input files, and stored as a contiguous **memory-mapped float array** plus an
optional usearch **HNSW** file. It is queried through a magic
`SERVICE <…/vectorSearch/>` clause that the planner turns into a
`VectorSearchJoin` operation — a near-clone of the existing `SpatialJoin`. With no
HNSW present, search is **exact brute force** (ideal for small databases); with
HNSW present, the planner chooses exact vs. approximate based on the **estimated
cardinality** of the search side, so a query that has already narrowed the
candidates to a few thousand entities runs an exact search instead of probing the
ANN index. Vectors are stored out-of-line because a `ValueId` has only 60 payload
bits — far too few for an embedding — the same reason geometries live in a side
file today.

> Status: **design proposal / no code yet.** See `integration-plan.md` §"Open
> decisions" in `architecture.md` for the choices to confirm before
> implementation (notably the v1 input format).
