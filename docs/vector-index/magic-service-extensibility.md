# Making magic `SERVICE`s isolated and extensible

> Investigation requested before the vector-index work: *"Is it possible to
> refactor QLever so addition of new `SERVICE`s is isolated and maintainable, so
> we don't need to mangle with existing core files — and possibly have them
> dynamically linked and discovered?"*
>
> Short answer: **Yes for compile-time isolation** — we can get adding a new
> magic service down to *"create 2–3 self-contained files + one registration
> line, with zero edits to core parser/planner files."* **Runtime dynamic
> loading (dlopen plugins) is technically possible but not recommended** for
> QLever as it stands; the reasons and the narrow path that *would* work are in
> §5.

This document is a prerequisite-design for the vector index: the
`VectorSearchJoin` should be the **first feature added through the new
mechanism**, proving it generalises. Companion docs: [`architecture.md`](./architecture.md),
[`integration-plan.md`](./integration-plan.md).

> **Status: implemented as a drop-in plugin architecture, 2026-06-16.**
> A custom magic `SERVICE` now lives entirely in its **own folder under
> `src/services/<name>/`** and wires itself in at runtime — **no edits to any core
> engine/parser file or CMake** to add another one. The pieces:
> - **Parser registry** `parser/MagicServiceRegistry` — services self-register an
>   IRI→factory; `SparqlQleverVisitor` has one generic branch and parses into a
>   single `parsedQuery::MagicService` variant node.
> - **Planner façade + registry** `engine/MagicServicePlanning.{h,cpp}` — a stable
>   `MagicServicePlanningContext` (hides `GraphPatternPlanner`/`makeSubtreePlan`)
>   plus a `MagicServicePlannerRegistry`. `QueryPlanner::planMagicService` is fully
>   generic; services register their planner handler from their own TU using only
>   the façade. **The planner no longer contains any per-service code.**
> - **Build** `src/services/CMakeLists.txt` globs service folders, builds each as
>   an OBJECT library (so self-registration survives the link), and links them into
>   the binaries. The top-level CMake has a single one-time `add_subdirectory(src/services)`.
>
> The **vector search** service is the worked example, fully relocated to
> `src/services/vectorSearch/` (parser query, config, the two operations, and a
> `VectorSearchService.cpp` that does both registrations). The core was cleaned of
> all vector references (variant, parser/planner dispatch, both `static_assert`s,
> the IRI constant). Verified end-to-end (all query forms) + regression
> SparqlParserTest 16/16, QueryPlannerTest 107/107, VectorIndexTest 6/6.
>
> **Index-extension decoupling done too (vector is now 100% in its folder).**
> `index/IndexExtension.{h,cpp}` adds a generic registry of **build hooks** (run
> by `qlever index` after the main index is built, with URI→Id resolution
> available) and **load hooks** (run at server start, after the vocab), plus a
> generic extension store on `IndexImpl` (`setExtension`/`getExtension`). The
> core index builder gained a single generic `--service-index '<json>'` option
> (a JSON object keyed by service name; each service picks its key). Vector's
> storage (`VectorIndex`/builder/reader), its `EmbeddingClient`, and its
> build/load hooks all moved into `src/services/vectorSearch/`; core retains only
> the generic registries. **Core has zero vector references** (verified). So a
> service *with its own on-disk index* is now also fully drop-in: register a build
> hook + a load hook (store via `setExtension`, read via `getExtension`).
>
> The other built-in services (spatial/text/path/…) were left on their dedicated
> variant alternatives (zero-risk); migrate incrementally if desired.

## 1. What "a magic SERVICE" is, and where it couples to core today

A magic service is a SPARQL `SERVICE <special-iri> { ... }` that QLever
interprets locally instead of contacting a remote endpoint (spatial search,
text search, path search, named-cached-result, materialized-view, external
values). Adding one today touches the following **core files** (verified
against the current tree):

| # | Seam | File / location | Why it's "core" |
| --- | --- | --- | --- |
| 1 | IRI constant | `src/parser/MagicServiceIriConstants.h:18-28` | central header |
| 2 | Parser dispatch | `src/parser/sparqlParser/SparqlQleverVisitor.cpp:1285-1304` — an `if/else` chain on the IRI calling `visitMagicServiceQuery<T>` | central visitor |
| 3 | **Parsed-query variant** | `src/parser/GraphPatternOperation.h:232-236` — a *closed* `std::variant` of 18 alternatives | every exhaustive `std::visit` must know all alternatives |
| 4 | Planner dispatch | `src/engine/QueryPlanner.cpp:3052-3150` — `graphPatternOperationVisitor`, a big `if constexpr` chain with one branch per service | central planner |
| 5 | Planner join hook (binary joins only) | `src/engine/QueryPlanner.cpp:2263-2401` — `createJoinCandidates` + `checkSpatialJoin`/`createSpatialJoin`, hard-coded to `SpatialJoin` | central planner |
| 6 | The service's own parser type | a `parsedQuery::XQuery : MagicServiceQuery` | self-contained ✓ |
| 7 | The service's own operation | an `XJoin/XScan : Operation` (+ algorithms) | self-contained ✓ |
| 8 | Build registration | `CMakeLists.txt` additions | unavoidable, trivial |

Seams 6–8 are already nicely isolated. **Seams 1–5 are the "mangling core
files" problem.** The root cause is seam #3: the parsed query is a *closed sum
type*, so every new service is a new variant alternative, which forces matching
changes in every place that switches over the variant (seams 2, 4, and the
exhaustive visitors).

Good news from the audit: the **only exhaustive visitor** over
`GraphPatternOperation` is the planner's `graphPatternOperationVisitor`
(reached via `child.visit(...)`, `QueryPlanner.cpp:277-278,3172-3173`). Every
other one of the ~30 files that mention the variant uses
`std::get`/`holds_alternative` on a *specific* alternative (mostly
`BasicGraphPattern`). So the closed variant is far less entangled than its
breadth suggests — collapsing the magic services out of it is tractable.

Also good: `MagicServiceQuery` (`src/parser/MagicServiceQuery.h:28-92`) is
*already* a polymorphic base with the right shape — `addParameter`, `addGraph`,
`validate`, `name`, and a `childGraphPattern_`. It is already 80% of a plugin
interface; today its subclasses are just *also* wrapped in the closed variant
and dispatched by hand.

## 2. Target design — a compile-time magic-service registry

The refactor turns the per-service hand-dispatch into **open registration
against two small interfaces**, removing seams 1–5 for all future services.

### 2.1 Collapse the N variant alternatives into ONE polymorphic node

Replace the six magic-service alternatives in the `GraphPatternOperation`
variant (`PathQuery, SpatialQuery, TextSearchQuery, NamedCachedResult,
MaterializedViewQuery, ExternalValuesQuery`) with a **single** alternative:

```cpp
// in parsedQuery::
struct MagicService {
  std::unique_ptr<MagicServiceQuery> query_;   // the polymorphic config object
};
```

`MagicServiceQuery` gains the few virtuals the core actually needs from a
service after parsing (it already has most):

```cpp
struct MagicServiceQuery {
  std::optional<GraphPattern> childGraphPattern_;
  virtual void addParameter(const SparqlTriple&) = 0;
  virtual void validate() const;
  virtual std::string_view name() const = 0;

  // NEW: the planner hook — turn this parsed config (+ already-planned child
  // subtrees) into a SubtreePlan. Each service implements its own planning.
  virtual SubtreePlan plan(QueryPlanner::GraphPatternPlanner& planner) const = 0;

  // NEW (optional): variables this service contributes, for the parser/analysis
  // passes that previously special-cased each alternative.
  virtual std::vector<Variable> visibleVariables() const { return {}; }
};
```

The single exhaustive planner branch becomes:

```cpp
} else if constexpr (std::is_same_v<T, p::MagicService>) {
  candidates_.push_back(arg.query_->plan(*this));   // virtual dispatch
}
```

— replacing the six hand-written `if constexpr` branches at
`QueryPlanner.cpp:3135-3147`. No planner edit is ever needed again for a new
service.

### 2.2 Parser side — an IRI→factory registry

Replace the `if/else` IRI chain (`SparqlQleverVisitor.cpp:1285-1304`) with a
lookup:

```cpp
// MagicServiceRegistry.h
class MagicServiceRegistry {
 public:
  using Factory = std::function<std::unique_ptr<MagicServiceQuery>(
      const TripleComponent::Iri& serviceIri)>;
  // match by exact IRI or by IRI prefix (some services use prefixes today)
  static void registerService(IriMatcher matcher, Factory factory);
  static std::optional<Factory> lookup(const TripleComponent::Iri&);
};
```

`visit(ServiceGraphPatternContext*)` then does: `if (auto f =
MagicServiceRegistry::lookup(iri)) return parseMagicService(ctx, *f);`,
falling through to the existing remote-`SERVICE` path otherwise. The generic
`visitMagicServiceQuery<T>` body (`SparqlQleverVisitor.cpp:1224-1260`) stays —
it already works for any `MagicServiceQuery` subclass; only the *selection* of
`T` moves into the registry.

Registration is one line per service, kept next to the service's own code:

```cpp
// in VectorSearchQuery.cpp
static const bool registered = (MagicServiceRegistry::registerService(
    ExactIri{VECTOR_SEARCH_IRI},
    [](auto&&) { return std::make_unique<VectorSearchQuery>(); }), true);
```

(Or, to avoid static-initialisation-order subtleties, an explicit
`registerBuiltinMagicServices()` called once at startup that lists the built-in
services — equally isolated, more predictable. Recommended.)

### 2.3 Planner join hook — an `IncompleteBinaryJoin` interface

For services that are *binary joins* assembled incrementally (spatial join
today, vector join tomorrow), generalise the hard-coded
`checkSpatialJoin`/`createSpatialJoin` (`QueryPlanner.cpp:2344-2401`) into one
interface implemented by the operations:

```cpp
class IncompleteBinaryJoin {
 public:
  virtual bool isConstructed() const = 0;
  virtual const Variable& leftJoinVar() const = 0;
  virtual const Variable& rightJoinVar() const = 0;
  virtual std::shared_ptr<Operation> addChild(
      std::shared_ptr<QueryExecutionTree>, const Variable&) const = 0;
};
```

`createJoinCandidates` (`:2263-2276`) then does a single
`dynamic_pointer_cast<const IncompleteBinaryJoin>` check and a generic
attach-child step, working for `SpatialJoin`, `VectorSearchJoin`, and any future
binary magic join with no further planner edits.

### 2.4 Net result

After this refactor, adding a magic service is:

- **New files only:** `XQuery.{h,cpp}` (parser config, implements `plan()`),
  `XOperation.{h,cpp}` (+ algorithms), and — if binary — implement
  `IncompleteBinaryJoin`.
- **One registration line** (in `registerBuiltinMagicServices()` or a static).
- **One CMake line.**
- **Zero edits** to `GraphPatternOperation.h`, `SparqlQleverVisitor.cpp`, or
  `QueryPlanner.cpp`.

That is the "isolated and maintainable" target, achieved entirely at compile
time with no ABI risk.

## 3. What this costs (one-time)

The collapse in §2.1 is itself a refactor of core files — but a *one-time* one
that pays for itself immediately (it removes the 6 existing per-service
branches) and forever after (every future service is zero-core-edit). The work:

1. Add the `plan()` virtual and move each existing service's planner branch
   (`QueryPlanner.cpp:3135-3147`) into its `XQuery::plan()`. Mechanical.
2. Introduce `p::MagicService`, migrate the 6 alternatives, update the single
   exhaustive visitor and the handful of `std::get<SpatialQuery>`-style sites
   (audit shows these are few and targeted) to go through the node.
3. Add `MagicServiceRegistry` and move the IRI `if/else` into registrations.
4. Extract `IncompleteBinaryJoin` from the spatial-join hook.

Each step is behaviour-preserving and independently testable; the existing
spatial/text/path/etc. tests are the safety net. This is best done as a small
**refactor phase that lands before (or alongside) the vector work**, so the
vector join is implemented *natively* against the new mechanism rather than
copy-pasting the old hand-dispatch and refactoring later.

## 4. Recommendation on ordering

Two viable orderings:

- **(A) Refactor-first.** Do §2 as a standalone, behaviour-preserving PR
  (migrating the existing services), then build the vector index purely against
  the registry. Cleanest end state; the vector PRs touch no core files. Slightly
  more up-front work.
- **(B) Vector-first, refactor-extract.** Implement `VectorSearchJoin` the way
  `SpatialJoin` is wired today (editing the 5 seams), then extract the registry +
  `IncompleteBinaryJoin` once two binary magic joins exist and the shared shape
  is concrete.

**Recommendation: (A)**, because the user's explicit goal is to *not* mangle core
files when adding services, and (A) is the only ordering that achieves that for
the vector feature itself. (B) is the pragmatic fallback if we want a working
vector index sooner and are willing to refactor afterwards.

## 5. The "dynamically linked and discovered" question — honest assessment

Could services be **shared-library plugins discovered at runtime** (e.g.
`dlopen` of a `.so` dropped in a directory), so a third party ships a vector or
custom service without recompiling QLever?

**Technically possible, but not recommended for QLever today.** The reasons are
structural, not incidental:

1. **QLever links everything statically.** The build produces static archives
   (`lib/*.a`) and one monolithic binary; there is no `BUILD_SHARED_LIBS`, no
   `MODULE`/`SHARED` target, no `-rdynamic`, and no symbol-visibility/export
   setup (verified). A plugin system is greenfield infrastructure, not a flag
   flip.
2. **There is no stable ABI.** The types a service must speak — `Operation`,
   `Result`, `IdTable`, `ValueId`, `QueryExecutionTree`, the allocator — are
   internal C++ classes that change frequently and are heavily templated
   (`CallFixedSize`, compile-time column widths, `if constexpr` visitors). C++
   has no cross-version/cross-compiler ABI guarantee; a plugin compiled against
   one QLever commit would silently break on the next. You would be maintaining
   an ABI contract over the engine's hottest, most volatile types.
3. **Templates can't cross a binary boundary.** Much of the engine's
   performance comes from monomorphised templates instantiated in the host. A
   plugin can only use what's behind a virtual interface; the fixed-width
   `IdTable` fast paths, etc., aren't reachable across `dlopen` without
   re-instantiation in the plugin (code bloat, version skew).
4. **Low value for the actual goal.** Dynamic discovery's payoff is a
   *third-party binary-plugin ecosystem*. Vector search is a first-class,
   in-tree feature of an open-source engine, not a third-party add-on. Recompiling
   to add a built-in service is normal and cheap relative to the fragility above.

**If runtime plugins are ever genuinely required**, the only sane path is a
**narrow, stable C ABI** at the boundary — not the C++ engine types:

- A versioned `extern "C"` entry point per plugin (`qlever_plugin_abi_version()`,
  `qlever_register(MagicServiceRegistryC*)`).
- Cross only POD / opaque-handle types and pure-virtual interfaces with a frozen
  vtable layout (an abstract `IMagicServiceQuery` / `IOperation` exposing just
  `computeResult`-shaped calls over opaque `IdTable*` handles).
- Accept the performance/feature ceiling that imposes (no template fast paths).

That is a substantial, separate project with ongoing ABI-maintenance cost. The
compile-time registry in §2 delivers ~all of the *maintainability* the user
asked for (isolated, no core-file edits per service) without any of it.

### Bottom line

- **Do:** the compile-time registry + `IncompleteBinaryJoin` extraction (§2).
  New services become self-contained files + one registration line. This is the
  right, low-risk answer to "isolated and maintainable."
- **Don't (for now):** runtime `dlopen` plugin discovery. Revisit only if a
  real third-party binary-plugin requirement appears, and then only behind a
  narrow frozen C ABI.

## 6. How the vector index uses this

Under ordering (A), the vector index integration plan's planner steps change as
follows (see [`integration-plan.md`](./integration-plan.md)):

- A new **Phase R** (refactor) precedes vector P1: implement §2 and migrate the
  existing services. Pure refactor, guarded by existing tests.
- Vector **P1/P3** then add `VectorSearchQuery` (implements `plan()` +
  registration), `VectorSearchJoin` (implements `IncompleteBinaryJoin`), and the
  algorithms — **with no edits to `GraphPatternOperation.h`,
  `SparqlQleverVisitor.cpp`, or `QueryPlanner.cpp`.** The vector feature becomes
  the proof that the mechanism generalises.
```
