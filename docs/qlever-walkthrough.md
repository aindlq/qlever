# QLever from the Ground Up — An Educational Walkthrough

> A tutorial for someone who has **never seen this codebase**. It explains what
> QLever is, how it turns an RDF file into a fast on-disk index, how it answers a
> SPARQL query, and finally how the **extension mechanism** (magic `SERVICE`s and
> index extensions — e.g. the built-in vector search) plugs into all of that
> *without touching the core engine*.
>
> You do **not** need to know SPARQL, RDF, or C++ template wizardry to follow the
> first half. Code identifiers and file paths are given so you can jump into the
> source, but the prose stands on its own. Line numbers drift as the code changes,
> so we mostly point at *files and class names*, which are stable.

---

## Table of contents

1. [What QLever is, in one picture](#1-what-qlever-is-in-one-picture)
2. [The foundation: everything is a 64-bit `ValueId`](#2-the-foundation-everything-is-a-64-bit-valueid)
3. [Build time: turning RDF text into an index](#3-build-time-turning-rdf-text-into-an-index)
4. [Run time: loading the index and reading from it](#4-run-time-loading-the-index-and-reading-from-it)
5. [The query engine: from SPARQL string to answer](#5-the-query-engine-from-sparql-string-to-answer)
6. [A worked example, traced end to end](#6-a-worked-example-traced-end-to-end)
7. [Extending QLever without forking it](#7-extending-qlever-without-forking-it)
8. [Case study: the built-in vector search](#8-case-study-the-built-in-vector-search)
9. [Map of the codebase](#9-map-of-the-codebase)

---

## 1. What QLever is, in one picture

QLever is a **SPARQL engine**: a database for *graph* data. The data is a set of
**triples** — little three-word sentences:

```
<Albert_Einstein>  <bornIn>   <Ulm> .
<Albert_Einstein>  <field>    <Physics> .
<Ulm>              <country>  <Germany> .
```

Each triple is **subject – predicate – object** (S, P, O). A whole knowledge
graph (Wikidata, for instance, has ~20 *billion* triples) is just an enormous
pile of such sentences. SPARQL is the query language for asking questions about
them ("give me everyone born in a German city who worked in physics").

QLever's whole life has **two phases**:

```
  ┌──────────────────────────────────────────────────────────────────┐
  │  PHASE 1 — BUILD (offline, run once with `qlever-index`)          │
  │                                                                    │
  │   data.ttl  ──parse──▶  triples  ──assign IDs──▶  sorted, packed   │
  │   (RDF text)            (S P O)    + vocabulary    binary files    │
  └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼  (the on-disk index)
  ┌──────────────────────────────────────────────────────────────────┐
  │  PHASE 2 — SERVE (online, `qlever-server` keeps running)          │
  │                                                                    │
  │   SPARQL query ──parse──▶ plan ──execute──▶ table of IDs ──▶ text  │
  └──────────────────────────────────────────────────────────────────┘
```

Two executables embody the two phases:

- **`qlever-index`** (entry point `src/index/IndexBuilderMain.cpp`) — reads RDF,
  writes the index files. You run it once.
- **`qlever-server`** — loads those files and answers queries over HTTP, forever.

Both are thin wrappers around the library in `src/libqlever/Qlever.{h,cpp}`.

The single most important idea, which makes everything else fall into place: **QLever
never works with strings during query execution. It works with integers.** Strings
are converted to 64-bit integer IDs *once*, at build time, and converted back to
text *once*, at the very end of a query. Everything in between — joins, filters,
sorting — is integer math. That is why it is fast, and it is the right place to
start.

---

## 2. The foundation: everything is a 64-bit `ValueId`

**File:** `src/global/ValueId.h`

Every value in QLever — an IRI like `<Ulm>`, a string literal, a number, a date,
a geographic point — is squeezed into a single 64-bit word called a `ValueId`
(aliased as `Id`). The 64 bits are split into two parts:

```
   63                                                              0
   ┌──────────┬──────────────────────────────────────────────────┐
   │  4 bits  │                    60 bits                        │
   │  type    │                    payload                        │
   └──────────┴──────────────────────────────────────────────────┘
     ▲                                ▲
     │                                └─ value, OR an index into a dictionary
     └─ which Datatype this is
```

In code: `numDatatypeBits = 4`, `numDataBits = 64 - 4 = 60`, and the largest
index that fits is `maxIndex = 2^60 - 1`.

The top 4 bits are a `Datatype` tag (`enum struct Datatype`). The datatypes fall
into two families:

**(a) "Trivial" / inline types — the value lives *right there* in the 60 bits:**

| Datatype    | What's in the payload                                    |
|-------------|---------------------------------------------------------|
| `Undefined` | nothing (this is SPARQL's UNDEF / "no value")           |
| `Bool`      | 0 or 1                                                   |
| `Int`       | a 60-bit signed integer                                  |
| `Double`    | a 64-bit double, shifted to make room for the tag (the low ~4 mantissa bits of precision are sacrificed) |
| `Date`      | a packed year/month/day/time encoding                   |
| `GeoPoint`  | a latitude/longitude point packed into exactly 60 bits  |

For these, decoding is pure arithmetic — no memory lookup. `42` really is stored
as `(Datatype::Int << 60) | 42`.

**(b) "Index" types — the 60 bits are a *pointer* (an offset) into a dictionary:**

| Datatype          | The 60 bits index into…                                  |
|-------------------|----------------------------------------------------------|
| `VocabIndex`      | the **vocabulary** (the big on-disk string dictionary)   |
| `LocalVocabIndex` | a small **per-query** vocabulary (strings *made* during a query, e.g. by `CONCAT`) |
| `TextRecordIndex` / `WordVocabIndex` | the full-text index                   |
| `BlankNodeIndex`  | a blank node                                             |

So `<Albert_Einstein>` is *not* stored as a string in the data tables. It is
stored once in the vocabulary at, say, position 8723, and everywhere it appears
in the graph it is just the `ValueId` `(VocabIndex << 60) | 8723`.

> **Teaching insight — why pack the type into the value?**
> Because comparison and sorting become a single 64-bit integer compare. The type
> tag sits in the *high* bits, so all values of the same type are automatically
> grouped together when you sort by raw integer value, and within a type the
> payload orders them. Joins, sorts, and range scans are then plain integer
> operations over contiguous memory — no pointer chasing, no string compares, no
> branching on type. This one trick is the reason QLever can join billions of rows
> quickly.

> **Teaching insight — why a separate vocabulary instead of inlining strings?**
> Strings are big and variable-length; IDs are 8 bytes and fixed. Storing the
> graph as fixed-width integers makes the data tables compact, cache-friendly, and
> sortable. The string ↔ ID translation happens exactly twice (in at build, out at
> the end of a query), never in the hot loop.

Keep this picture in your head. The rest of QLever is "how do we build the
dictionary and the integer tables" (Part 3) and "how do we compute over the
integer tables" (Part 5).

---

## 3. Build time: turning RDF text into an index

**Entry point:** `src/index/IndexBuilderMain.cpp` → `qlever::Qlever::buildIndex`
→ `IndexImpl::createFromFiles` (`src/index/IndexImpl.cpp`).

You invoke it roughly like:

```
qlever-index -i mydb -f data.ttl -F ttl
```

(`-i` = output basename, `-f` = input RDF file, `-F` = format.) Out of this comes
a handful of binary files all named `mydb.*`. Here is what happens, in order.

### Stage 1 — Parse the RDF into triples

The RDF text is tokenized and parsed by the `RdfParser` family
(`src/parser/RdfParser.h`; `TurtleParser` handles Turtle/N-Triples/N-Quads). The
output is a stream of triples whose components are `TripleComponent` objects —
each is either an IRI, a blank node, or an already-recognized literal.

Crucially, literals with a known datatype are encoded **inline immediately**:
`"42"^^xsd:integer` becomes the `Int` `ValueId` `42`; a date literal becomes a
`Date` `ValueId`. Only things that need the dictionary (IRIs, plain strings) are
left to the vocabulary stage.

> **Why parse in batches?** Big dumps don't fit in RAM. QLever parses in chunks
> (millions of triples at a time), and several stages below are *external*
> (disk-backed) merge-sorts for the same reason.

### Stage 2 — Build the vocabulary (strings → `VocabIndex`)

Files: `src/index/VocabularyMerger.h`, `src/index/vocabulary/*`.

This is a classic external-sort in three passes:

1. **Partial vocabularies.** Each batch of parsed triples produces a *partial*
   vocabulary (a sorted set of the unique strings it saw) written to temp files,
   plus a map from "string → temporary local id".
2. **Merge.** All partial vocabularies are merge-sorted into **one** globally
   sorted vocabulary on disk (suffix `.vocabulary`). Sorting the strings means a
   string's final `VocabIndex` is simply its rank in sorted order — which later
   lets QLever do *range* queries on strings and binary-search lookups.
3. **Re-map.** A final pass rewrites every triple's temporary ids into the global
   `VocabIndex` ids, and splits triples into "normal" (your data) vs "internal"
   (QLever's own bookkeeping, see patterns below).

QLever also distinguishes an **internal** vocabulary (its own reserved IRIs, and
optionally small/hot values kept in RAM) from the **external** vocabulary (the
bulk, on disk, possibly compressed). That split lets the engine special-case its
own predicates and keep the giant user dictionary out of precious RAM.

### Stage 3 — Build the six permutations

Files: `src/index/Permutation.h`, `src/index/CompressedRelation*`.

Now we have triples as `(Id, Id, Id)`. A SPARQL query can fix any subset of the
three positions and ask for the rest (`<Einstein> <bornIn> ?x`, or `?s ?p <Ulm>`,
or `?s <bornIn> ?o`). To make *every* such pattern an indexed lookup instead of a
full scan, QLever stores the triples sorted **six** different ways — one per
ordering of S, P, O:

```
   enum struct Enum { PSO, POS, SPO, SOP, OPS, OSP };
```

Think of it like a phone book. A book sorted by *last name* answers "find Smith"
instantly but is useless for "who lives on Oak St." If you keep the same entries
sorted six ways, then whichever fields the query fixes, there is always a sorted
copy where those fields are the leading prefix, so the matching rows form one
contiguous, binary-searchable run.

| If the query fixes… | use the permutation whose prefix matches |
|---------------------|------------------------------------------|
| predicate (`?s <p> ?o`) | PSO or POS                            |
| subject (`<s> ?p ?o`)   | SPO or SOP                            |
| object (`?s ?p <o>`)    | OPS or OSP                            |

Each permutation is **not** stored as fat 24-byte rows. It is stored
**column-wise**, cut into blocks, and each block is **compressed** (the leading
columns are extremely repetitive after sorting, so they shrink enormously).
Alongside the compressed data, QLever writes **block metadata**
(`CompressedBlockMetadata`): for each block it records the first and last triple
it contains. That metadata is the index *into* the index — to find a value, you
binary-search the tiny metadata to find the one block that could contain it, then
decompress only that block.

> **Teaching insight — why six and not three?** Three permutations would cover
> every *single-position* lookup, but joins and two-position patterns need the
> right *secondary* sort order too. Six permutations guarantee that for any bound
> pattern the answer is a single sorted range — turning would-be full scans into
> `O(log n)` seeks. The cost is disk space, which compression largely pays back.
> (You can build only PSO+POS with `--only-pso-and-pos-permutations` if you never
> query by subject/object alone.)

### Stage 4 — The "pattern trick" (an optimization)

File: `src/index/PatternCreator.h`.

Many queries ask "what predicates does this subject have?" (think: rendering all
properties of an entity). QLever precomputes, for each subject, its *set* of
predicates — its **pattern** — and stores compact `ql:has-pattern` relationships
in the **internal** PSO/POS permutations. This makes certain counting/grouping
queries dramatically cheaper and feeds the planner's cardinality estimates. It's
optional (`--no-patterns` turns it off).

### What ends up on disk

For basename `mydb`, a build writes exactly the files below. (These are the real
names — verified against built indexes; the naming logic is in
`Permutation::loadFromDisk`/the constructor in `src/index/Permutation.cpp`, where
`fileSuffix_ = "." + lowercase(toString(permutation))` and the metadata sidecar
appends `MMAP_FILE_SUFFIX = ".meta"`. Suffix constants are in
`src/global/Constants.h`.)

**Manifest**
- `mydb.meta-data.json` — configuration/manifest (`CONFIGURATION_FILE`): format
  version, counts, locale, settings. Read first on load.

**Vocabulary** (the string dictionary; `VOCAB_SUFFIX = ".vocabulary"`). The exact
companion files depend on the vocabulary implementation — a compressed vocabulary
produces:
- `mydb.vocabulary.words.external` + `mydb.vocabulary.words.external.offsets`
- `mydb.vocabulary.words.internal` + `mydb.vocabulary.words.internal.ids`
- `mydb.vocabulary.codebooks` (compression codebooks)

  If a `GeoVocabulary` is present, it adds a parallel `mydb.vocabulary.geometry.*`
  set, including the out-of-line satellite file `mydb.vocabulary.geometry.geoinfo`
  (see Part 4).

**The six permutations** — each a data file plus a memory-mapped block-metadata
sidecar (`.meta`):
- `mydb.index.pso` + `mydb.index.pso.meta`
- `mydb.index.pos` + `mydb.index.pos.meta`
- `mydb.index.spo` + `mydb.index.spo.meta`
- `mydb.index.sop` + `mydb.index.sop.meta`
- `mydb.index.ops` + `mydb.index.ops.meta`
- `mydb.index.osp` + `mydb.index.osp.meta`

**Internal permutations** (the pattern-trick / `ql:has-pattern` triples — only
PSO and POS, since `Permutation::INTERNAL = {PSO, POS}`):
- `mydb.internal.index.pso` + `mydb.internal.index.pso.meta`
- `mydb.internal.index.pos` + `mydb.internal.index.pos.meta`

**Patterns**
- `mydb.index.patterns` (omitted with `--no-patterns`)

Plus `mydb.text.*` files if you also build a text index.

That's the whole index. Note what's *not* there: no engine, no query. The index
is a passive, sorted, compressed data structure. Bringing it to life is Part 4
and 5.

---

## 4. Run time: loading the index and reading from it

**Entry point:** `IndexImpl::createFromOnDiskIndex` (`src/index/IndexImpl.cpp`),
called when `qlever-server` starts.

### Loading: mmap the metadata, stream the data

Loading is fast and lazy by design. In order, it:

1. Reads `mydb.meta-data.json` (validates the format version, restores settings).
2. Opens the **vocabulary** so IDs can be turned back into strings.
3. Runs any **index-extension load hooks** (this is where the vector index gets
   loaded — see Part 7).
4. Opens the **permutations**: it **memory-maps the small block-metadata** and
   opens the big compressed data files for on-demand reading. PSO/POS (which carry
   the internal/pattern data) load first; the others load if present.
5. Loads patterns and any persisted updates.

The recurring trick is **memory mapping** via `ad_utility::MmapVector` /
`MmapVectorView` (`src/util/MmapVector.h`). An `MmapVector<T>` is an STL-like array
whose storage *is a file*: you access `vec[i]` and the OS pages the right bytes in
from disk on demand, caching them in RAM transparently. A small trailer in the
file (with a magic number for validation) records the size/capacity.

> **Teaching insight — what gets mmap'd vs read?** The *metadata* (block
> boundaries — small, hot, touched on every scan) is memory-mapped so it lives in
> the OS page cache for near-instant access. The *compressed bulk data* (huge,
> read sequentially, decompressed once) stays on disk and is streamed block by
> block. You get instant startup and a small RAM footprint even for a
> billion-triple index, because you never "load the database into memory" — you
> let the kernel's virtual-memory system be your cache.

### Reading: a scan produces an `IdTable`

The unit of "reading from the index" is a **scan** of one permutation:
`Permutation::scan(...)` (`src/index/Permutation.cpp`), which delegates to the
`CompressedRelationReader`. Given a scan specification (which positions are fixed
to which IDs), it:

1. binary-searches the mmap'd block metadata to find the blocks that *could* match,
2. decompresses just those blocks,
3. filters to the matching rows,
4. returns them as an **`IdTable`** — a 2-D table of `ValueId`s (Part 5).

That `IdTable` of integer IDs is the raw material the query engine joins, filters,
and aggregates.

### Out-of-line satellite storage (the pattern our extensions reuse)

Sometimes a value needs *more* data than fits in its 60-bit `ValueId` — e.g. a
geometry literal needs an actual polygon, and (foreshadowing) a vector-search
entity needs a 768-float embedding. QLever's answer is the **`GeoVocabulary`**
pattern (`src/index/vocabulary/GeoVocabulary.h`): keep the main vocabulary as
usual, and keep a **separate, fixed-width side file** of the heavy data — the real
file is `mydb.vocabulary.geometry.geoinfo` (an 8-byte version header followed by
fixed-size `GeometryInfo` records) — indexed by the same `VocabIndex`. Looking up
entry *i*'s geometry is just `seek(header + i * recordSize)` — `O(1)`, no scanning.

> **Teaching insight — why out-of-line, fixed-width side files?** Not every value
> has the heavy data, and the heavy data is large. Inlining it would bloat the
> dictionary and ruin its cache behavior. A parallel fixed-width file gives
> constant-time lookup by index while keeping the main structures lean. **This is
> exactly the template the vector index follows** (a flat mmap'd float store keyed
> by `ValueId`), which is why the vector feature could be added without inventing
> new storage machinery.

---

## 5. The query engine: from SPARQL string to answer

A query travels through three stages — **parse → plan → execute** — plus caching
and export. Here they are.

### 5.1 Parse: SPARQL text → `ParsedQuery`

**Files:** grammar `src/parser/sparqlParser/generated/SparqlAutomatic.g4`
(ANTLR4), visitor `src/parser/sparqlParser/SparqlQleverVisitor.{h,cpp}`, result
type `src/parser/ParsedQuery.h`.

ANTLR turns the query text into a parse tree; the `SparqlQleverVisitor` walks that
tree and builds a `ParsedQuery` — a structured, engine-friendly representation
holding the `SELECT`/`CONSTRUCT`/… clause, `ORDER BY`, `GROUP BY`,
`LIMIT`/`OFFSET`, and most importantly the **`WHERE` clause** as a tree of
graph-pattern operations.

The heart of it is `parsedQuery::GraphPatternOperation`
(`src/parser/GraphPatternOperation.h`), a **`std::variant`** — a tagged union —
over every kind of pattern SPARQL allows:

```
std::variant< BasicGraphPattern,  // a block of triples
              Optional, Union, Minus, GroupGraphPattern,
              Bind, Values, Subquery, Service,
              ... ,
              MagicService >       // ← the generic extension node (Part 7)
```

So a SPARQL `WHERE` clause becomes a tree of these variant nodes. The parser's
job is *only* to faithfully represent the query; it makes no decisions about how
to run it.

> **Teaching insight — why a variant, not a class hierarchy?** The node kinds are
> a small, fixed, closed set. A `variant` makes "handle every case" a
> *compile-time-checked* `std::visit` — forget a case and it won't compile — with
> no virtual-dispatch or heap indirection. (Note the last alternative,
> `MagicService`: that's the single generic hook that lets *new* services be added
> without ever editing this list. Hold that thought for Part 7.)

### 5.2 Plan: `ParsedQuery` → `QueryExecutionTree`

**File:** `src/engine/QueryPlanner.{h,cpp}`.

This is the brain. The same query can be executed many ways — which triple to scan
first, which join order, whether to sort now or later — and they differ in cost by
*orders of magnitude*. The planner's job is to choose a good one.

Key concepts:

- A **`SubtreePlan`** wraps a candidate `QueryExecutionTree` plus bookkeeping
  (which triples/filters it already covers, and its estimated cost & size).
- The planner seeds **leaf** plans: each triple pattern becomes one or more
  `IndexScan`s (one per usable permutation).
- It then combines plans **pair by pair** with `Join`s, exploring orders via
  **dynamic programming** — building the cheapest plan for every subset of triples
  and growing those into the cheapest plan for the whole pattern.
- "Cheapest" is decided by **cost and cardinality estimates** —
  `getCostEstimate()`, `getSizeEstimateBeforeLimit()`, `getMultiplicity()` — which
  draw on the index's statistics (and the pattern data from Stage 4). Plans that
  are dominated get pruned.

The output is a single **`QueryExecutionTree`** (`src/engine/QueryExecutionTree.h`):
a tree whose nodes are **operations**.

> **Teaching insight — why estimate at all?** Picking the join order is *the*
> performance lever in a database. Scanning the 5 "green statues" and joining them
> into a billion-row relation is trivial; doing it the other way around melts your
> machine. The estimates are how the planner tells those apart before running
> anything. (This same cardinality awareness is what lets the vector service choose
> exact brute-force over a small candidate set vs. the approximate HNSW index.)

### 5.3 Execute: the operation tree computes `IdTable`s

**Files:** base class `src/engine/Operation.h`; data `src/engine/idTable/IdTable.h`
and `src/engine/Result.h`.

Every node in the execution tree is an `Operation`. The abstract base defines a
small contract that *every* operation implements:

| Method | Meaning |
|--------|---------|
| `computeResult(requestLaziness)` | **do the work**; produce the rows |
| `getResultWidth()` | how many columns the output has |
| `getCacheKeyImpl()` | a string that *uniquely identifies this subtree* (for caching) |
| `computeVariableToColumnMap()` | which SPARQL variable lives in which column |
| `getSizeEstimateBeforeLimit()` / `getCostEstimate()` / `getMultiplicity()` | the estimates the planner uses |
| `getChildren()` | child operations (so the tree can be traversed) |
| `cloneImpl()` | copy the operation |

Concrete operations (in `src/engine/`) include:

- **`IndexScan`** — a leaf; scans a permutation (Part 4) and emits matching triples.
- **`Join`** — merge-join two children on a shared variable/column.
- **`OptionalJoin`** — SPARQL `OPTIONAL` (left outer join; missing values become UNDEF).
- **`Filter`** — keep rows where a SPARQL expression is true.
- **`Sort`** (raw ID order, to prepare joins) and **`OrderBy`** (semantic `ORDER BY`).
- **`Distinct`**, **`GroupBy`** (aggregation), **`Bind`** (compute a new column),
  **`Union`**, **`CartesianProductJoin`**, **`Service`** (federate to a remote
  SPARQL endpoint), text-search operations, and more.

Execution is **demand-driven and bottom-up**: you ask the root for its result, it
asks its children, down to the `IndexScan` leaves, and `IdTable`s flow back up.

**The data that flows:**

- An **`IdTable`** is a 2-D array of `ValueId`s, stored **column-major** (each
  column contiguous). Column-major because operations usually touch one or two
  columns (the join key, a filtered value), so keeping a column contiguous is
  cache-friendly and SIMD-friendly.
- A **`Result`** wraps the output. It can be **fully materialized** (one big
  `IdTable`) or **lazy** — a coroutine **generator** that yields the table in
  chunks. Laziness lets a huge intermediate result stream through `LIMIT`/filters
  without ever existing all at once, and lets a cancelled query stop promptly.
- A **`LocalVocab`** (`src/index/LocalVocab.h`) rides along with the result. It
  holds strings *created during the query* (e.g. by `CONCAT` or `STR`) that aren't
  in the on-disk vocabulary; these get `LocalVocabIndex` IDs so they behave like
  any other value for the rest of the pipeline.

### 5.4 Caching

**File:** `src/engine/QueryExecutionContext.h` (the `QueryResultCache`).

Because `getCacheKeyImpl()` gives every subtree a unique string key, QLever caches
**results of sub-expressions**, not just whole queries. Two different queries that
share a common sub-pattern (say, the same expensive scan-and-join) reuse the same
cached `IdTable`. Before computing, an operation checks the cache for its key; on a
hit it returns the stored result.

> **Teaching insight — why string cache keys?** A string key is a complete,
> portable, human-readable fingerprint of an arbitrary subtree. It makes
> sub-result reuse across *different* queries trivial (just compare keys) and makes
> the cache easy to inspect and debug — at the cost of building a string, which is
> negligible next to executing the operation.

### 5.5 Export: IDs → text

Finally `ExportQueryExecutionTrees` (`src/engine/`) walks the root result and
turns `ValueId`s back into text — looking each one up in the vocabulary (or
`LocalVocab`, or decoding it inline) and formatting per the requested media type
(CSV, TSV, SPARQL-JSON, QLever-JSON, Turtle, binary). It streams the output in
chunks, applying any final `LIMIT`/`OFFSET`. This is the *second and last* time
strings enter the picture.

---

## 6. A worked example, traced end to end

Query:

```sparql
SELECT ?person WHERE {
  ?person <bornIn> ?city .
  ?city   <country> <Germany> .
}
```

1. **Parse.** Two triple patterns land in a `BasicGraphPattern` inside the
   `ParsedQuery`'s root graph pattern. `?person`, `?city` are variables;
   `<bornIn>`, `<country>`, `<Germany>` will be looked up as `VocabIndex` IDs.

2. **Plan.** The planner seeds leaves:
   - `?person <bornIn> ?city` → `IndexScan` on **PSO** (predicate `<bornIn>` fixed)
     → a 2-column `IdTable` (person, city).
   - `?city <country> <Germany>` → `IndexScan` on **POS** (predicate `<country>`
     and object `<Germany>` fixed) → a 1-column `IdTable` (city).

   It estimates: the second scan is tiny (few German-country cities? actually many
   cities, but a specific country), the first is large. It plans a **`Join`** on
   the shared `?city` column, choosing the cheaper order, and adds a `Sort` if a
   side isn't already sorted on the join column. Result: a `QueryExecutionTree`
   like `Join(IndexScan_PSO, IndexScan_POS)`.

3. **Execute.** Root `Join::computeResult()` pulls both children's `IdTable`s
   (each `IndexScan` decompresses the relevant blocks of its permutation),
   merge-joins them on the `?city` column, and emits an `IdTable` with the
   surviving `?person` IDs. All integer operations.

4. **Cache.** Each subtree's result is stored under its cache key; a later query
   reusing `?city <country> <Germany>` skips recomputation.

5. **Export.** Each `?person` `ValueId` is resolved through the vocabulary back to
   an IRI string and streamed out as CSV/JSON/etc.

Everything between step 1 and step 5 is integer math over compressed, sorted,
memory-mapped tables. That's QLever.

---

## 7. Extending QLever without forking it

Now the part this repository adds. The goal was: **add entirely new query
capabilities (vector similarity search today; others tomorrow) as self-contained
modules that live in their own folder and require **zero edits** to the core
engine, parser, or planner.** That makes the "fork" trivially rebasable on
upstream QLever — your code sits beside the engine, not inside it.

There are two things a rich feature needs to hook into, and QLever now exposes a
clean seam for each: **the query language** (a new `SERVICE`) and **the index**
(extra on-disk data). Both use the same idea: a **registry** that modules
self-register into at startup, so the core calls them generically.

### 7.1 Magic `SERVICE`s — extending the query language

SPARQL has a `SERVICE` keyword (normally "federate this sub-query to a remote
endpoint"). QLever repurposes `SERVICE <special-iri> { ... }` as the syntax for
**built-in** special operations — spatial joins, text search, and now vector
search. These are "magic services". The extension machinery has four small pieces:

**(1) One generic variant node.** Instead of adding a new alternative to the
`GraphPatternOperation` variant for every service (which would edit core code each
time), there is a *single* generic `MagicService` node
(`src/parser/GraphPatternOperation.h`) holding a
`std::shared_ptr<MagicServiceQuery>`. Every service's parsed form is a subclass of
`MagicServiceQuery`. The variant — and the exhaustive `std::visit`s over it —
never change again.

**(2) A parser registry.** `parser/MagicServiceRegistry`
(`src/parser/MagicServiceRegistry.{h,cpp}`). A service registers *(a)* a matcher
that recognizes its special IRI and *(b)* a factory that builds its
`MagicServiceQuery` subclass:

```cpp
// In the service's own .cpp, at static-init time:
MagicServiceRegistry::get().addExact(VECTOR_SEARCH_IRI, [](const Iri&) {
  return std::make_shared<VectorSearchQuery>();
});
```

The SPARQL visitor has **one** generic branch: when it sees a `SERVICE` whose IRI
the registry recognizes, it builds a `MagicService` node via the factory. No
per-service code in the parser.

**(3) A planner registry + a façade.** `engine/MagicServicePlanning.{h,cpp}`. The
planner (`QueryPlanner.cpp`) has **one** generic handler, `planMagicService`. It
looks up a planner function by the *type* of the parsed query object
(`std::type_index(typeid(query))`) and calls it. The service's planner builds its
operations through a small **`MagicServicePlanningContext`** façade — a stable
public interface (`qec()`, `addLeafOperation(...)`,
`addOperationWithChildPattern(...)`) that hides the planner's private internals
(`makeSubtreePlan`, `optimize`, …). So a service can do real planning (including
planning a *nested* graph pattern) from its own translation unit without touching,
or even including, the planner's guts.

**(4) The operations themselves** are ordinary `Operation` subclasses (Part 5.3).
For vector search these are `VectorSearch` (a leaf) and `VectorSearchJoin` (a
binary "for each row, search" op). At execution time they are dispatched exactly
like `Join` or `Filter` — there is **nothing special** about them in the hot loop.

> **Where the registries are consulted — and the performance consequence.** The
> parser registry is hit **once per `SERVICE` clause at parse time** (a tiny linear
> scan of matchers). The planner registry is hit **once per `SERVICE` clause at
> plan time** (one hash lookup by `type_index` + one `std::function` call through
> the façade). **Neither is ever touched during execution.** The operations run on
> the same virtual-dispatch path as every native operation, so there is **zero
> per-row overhead** — a magic service costs exactly what a built-in operation
> costs. And because everything is statically linked into one binary (see 7.3),
> there is no `dlopen`, no shared-library indirection, no ABI boundary.

### 7.2 Index extensions — attaching your own on-disk data

A feature like vector search needs its *own* files (the embeddings + the ANN
index) built at index time and loaded at server start. That's what
`src/index/IndexExtension.{h,cpp}` provides: a registry of **hooks**.

```cpp
using BuildHook = std::function<void(const Index&, const std::string& basename,
                                     const nlohmann::json& config)>;
using LoadHook  = std::function<void(IndexImpl&, const std::string& basename)>;
```

- **Build hooks** run **after the main index is built**, from
  `Qlever::buildIndex` (`src/libqlever/Qlever.cpp`). At that point the vocabulary
  exists, so the hook can **resolve IRIs → IDs** — essential, because the vector
  store is keyed by the same `ValueId`s the engine uses. The hook reads its slice
  of a generic `--service-index '<json>'` config and writes its own `basename.*`
  files.
- **Load hooks** run during `IndexImpl::createFromOnDiskIndex` **right after the
  vocabulary is loaded** (Part 4), memory-map the service's files, and stash the
  result on the index via a generic, type-erased store:

```cpp
// On IndexImpl: a generic bag of named auxiliary data.
ad_utility::HashMap<std::string, std::shared_ptr<void>> extensions_;
void setExtension(std::string name, std::shared_ptr<void> data);
std::shared_ptr<void> getExtension(const std::string& name) const;
```

The core stores `shared_ptr<void>` — it has **no compile-time knowledge** of what a
vector index is. The service casts it back to its real type when it needs it.

> **Teaching insight — why hooks run *after* the vocabulary.** The whole point of
> an extension's data is to connect to the graph's entities, and entities are
> `ValueId`s derived from the vocabulary. Running build hooks after the main build
> (and load hooks after the vocab loads) guarantees the URI↔ID mapping is final and
> available. It's the same reason the `GeoVocabulary` side file is keyed by
> `VocabIndex`: extensions speak the engine's native ID language.

### 7.3 Why this needs no core CMake edits either

Self-registration relies on static initializers running. But a static library only
links in the object files whose symbols are referenced — and nothing in the core
*references* a plugin, so its registration code would be silently dropped. The fix
(`src/services/CMakeLists.txt`): each service is built as a CMake **OBJECT
library** and linked *whole* into the binaries, so its static initializers survive.
A single top-level `add_subdirectory(src/services)` globs every service folder.
Adding a service is therefore: **drop a folder in `src/services/`** — no edits to
any core `CMakeLists.txt`.

### 7.4 Checklist: adding your own `SERVICE`

To add a brand-new magic service `foo`, create `src/services/foo/` containing:

1. A `FooQuery : MagicServiceQuery` (its parsed form) whose `.cpp`
   **self-registers** a parser factory in `MagicServiceRegistry` for your special
   IRI.
2. One or more `Operation` subclasses implementing the actual computation.
3. A registration of a **planner function** in `MagicServicePlannerRegistry` that
   builds those operations via the `MagicServicePlanningContext` façade.
4. *(Only if you need on-disk data)* register **build/load hooks** in
   `IndexExtensionRegistry`, and store/fetch your loaded data with
   `setExtension`/`getExtension`.
5. A `CMakeLists.txt` that builds an OBJECT library (the `src/services` glue links
   it in automatically).

That's it. You never edit the parser, the planner, the variant, the index core, or
any core CMake file. Your module rebases cleanly on top of upstream QLever.

---

## 8. Case study: the built-in vector search

Putting Part 7 together with a concrete feature (code in
`src/services/vectorSearch/`; design notes in `docs/vector-index/`). Vector search
lets you store an **embedding** (a list of floats) per entity and ask for the
*k nearest* entities to a query vector — and **join that with normal SPARQL**.

**At build time** (build hook, `VectorIndexExtension.cpp`): given
`--service-index '{"vectorSearch":[{"name":"clip","npy":…,"iris":…,"metric":"cosine","hnsw":true}]}'`,
the hook loads the precomputed vectors, **resolves each entity IRI to its
`ValueId`** via the just-built vocabulary, and writes these files (real names,
from `src/services/vectorSearch/VectorIndexFormat.h`; here basename `mydb`, index
name `clip`):

- `mydb.vec.clip.meta` — JSON metadata (dimensions, metric, config).
- `mydb.vec.clip.keys` — `MmapVector<uint64_t>`: the entity IDs (`ValueId`s),
  ascending, so entity→row is a binary search.
- `mydb.vec.clip.data` — the flat row-major float matrix (row *i* is the vector
  of entity `keys[i]`) — the `GeoVocabulary` satellite-file pattern from Part 4.
- `mydb.vec.clip.hnsw` — the usearch **HNSW** approximate-nearest-neighbor index
  (only if `hnsw` was requested).

(It can even *generate* embeddings at build time by calling an OpenAI-compatible
`/v1/embeddings` endpoint — see `docs/vector-index/embedding-generation.md`.)

**At load time** (load hook): the `basename.vec.*` files are memory-mapped and
attached with `setExtension("vectorSearch", …)`. Huge stores cost almost no RAM —
the OS pages vectors in as searched (Part 4).

**At query time:**

```sparql
SELECT ?img ?score WHERE {
  SERVICE <…/vectorSearch/> {
    _:c vec:index "clip" ;
        vec:queryText "a red bicycle" ;   # embedded on the fly via the endpoint
        vec:result ?img ;
        vec:k 10 ;
        vec:bindScore ?score .
  }
  ?img <license> <CC0> .                  # ← joined with the graph!
}
```

- The parser recognizes the special IRI (parser registry) and builds a
  `VectorSearchQuery`.
- The planner's generic `planMagicService` finds the registered vector handler,
  which builds a `VectorSearch` operation (or a `VectorSearchJoin` for the
  "for each ?x, search" form) through the façade.
- `computeResult` fetches the store via `getExtension`, runs the search (exact
  brute force for a small candidate set — **cardinality-aware**, just like the
  planner's join decisions — or HNSW for large ones), and emits an `IdTable` of
  result `ValueId`s + score columns.
- From there it's an ordinary `IdTable` that joins with `?img <license> <CC0>`
  using the same `Join` machinery as everything else.

No core file was edited to make any of this exist. That is the payoff of Part 7.

---

## 9. Map of the codebase

A starter map for exploring the source:

| Area | Where | Read this for… |
|------|-------|----------------|
| ID encoding | `src/global/ValueId.h`, `Id.h` | the 4+60-bit value model |
| Index build | `src/index/IndexBuilderMain.cpp`, `IndexImpl.cpp` | the build pipeline |
| Vocabulary | `src/index/VocabularyMerger.h`, `src/index/vocabulary/` | string↔ID dictionary |
| Permutations | `src/index/Permutation.h`, `CompressedRelation*` | the six sorted, compressed stores |
| Out-of-line data | `src/index/vocabulary/GeoVocabulary.h` | the satellite-file pattern |
| Mmap storage | `src/util/MmapVector.h` | file-backed arrays |
| Parsing | `src/parser/sparqlParser/SparqlQleverVisitor.*`, `ParsedQuery.h`, `GraphPatternOperation.h` | SPARQL → `ParsedQuery` |
| Planning | `src/engine/QueryPlanner.{h,cpp}` | join ordering, cost model |
| Execution | `src/engine/Operation.h`, `QueryExecutionTree.h` | the operation contract |
| Data | `src/engine/idTable/IdTable.h`, `Result.h`, `src/index/LocalVocab.h` | tables of IDs flowing through |
| Caching | `src/engine/QueryExecutionContext.h` | sub-result reuse |
| Export | `src/engine/ExportQueryExecutionTrees.*` | IDs → text |
| **Service extension** | `src/parser/MagicServiceRegistry.*`, `src/engine/MagicServicePlanning.*` | adding a `SERVICE` |
| **Index extension** | `src/index/IndexExtension.*`, `IndexImpl` `extensions_` | adding on-disk data |
| **Vector case study** | `src/services/vectorSearch/`, `docs/vector-index/` | the whole pattern, applied |

---

### One-paragraph summary

QLever converts RDF strings into 64-bit `ValueId`s (4 type bits + 60 payload), so
that all query work is integer math over compressed, sorted, memory-mapped tables.
The **index builder** writes a sorted string vocabulary plus six permutations of
the triples (one per S/P/O ordering, so any pattern is an indexed range scan). The
**server** memory-maps that index and answers queries in three stages —
**parse** (SPARQL → a `ParsedQuery` whose `WHERE` is a variant tree of graph
patterns), **plan** (a cost/cardinality-driven dynamic-programming search for a
good `Operation` tree), and **execute** (demand-driven `Operation`s passing
`IdTable`s of IDs up the tree, cached by string keys, finally exported back to
text). New capabilities plug in through **registries** — a magic `SERVICE` (parser
+ planner registry + a generic variant node + a planning façade) and an **index
extension** (build/load hooks + a type-erased `extensions_` store) — so a feature
like the built-in **vector search** lives entirely in `src/services/` and adds
itself at startup with **zero edits to the core engine**.
