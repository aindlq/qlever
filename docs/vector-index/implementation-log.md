# Vector index — implementation log

A running, honest record of executing the [integration plan](./integration-plan.md):
what was done, what was verified, and every issue hit + how it was resolved.

Newest entries at the bottom of each phase. Dates are UTC.

---

## 2026-06-16 — Phase 0 prep: establish a build/verify loop

**Intent:** before writing feature code, confirm I can compile and test changes
in this checkout.

### Issue #1 (BLOCKER) — this checkout cannot build or run QLever here

While probing the toolchain I found the build environment is **not present** in
this machine:

- The configured `build/` was generated for a *different* path
  (`/home/artem/work/projects/pharos/github/qlever`), which no longer exists.
  CMake builds are not relocatable, so `build/compile_commands.json` and
  `CMakeCache.txt` point at stale paths.
- The original build used a **Nix devshell**. Its toolchain is gone: the
  compiler `…/nix/store/…-gcc-wrapper-14.3.0/bin/g++` does not exist.
- The external dependencies are gone: `Boost` and `ICU` (and `OpenSSL`, `zstd`)
  were provided from `/nix/store/…` paths that no longer exist. There is **no
  `nix`/`nix-shell`**, **no `conan`/`vcpkg`**, and **no system `-dev` packages**
  for boost/icu/openssl/zstd. QLever needs all of these (Boost.Asio for the HTTP
  server, Boost.Geometry, ICU for the vocabulary, …).
- The prebuilt `build/ServerMain` / `IndexBuilderMain` (≈1 GB each) **do not
  run** — their ELF interpreter is a now-missing nix-store path.

What *is* available: system **g++ 13.3.0 + STL**, and the already-fetched
in-tree dependency sources under `build/_deps/` (abseil, antlr, ctre, fsst,
googletest, nlohmann-json, range-v3, re2, s2, spatialjoin). Network works at
configure time.

**Consequence:** I cannot compile/link/test the full QLever engine (parser,
planner, operations, index) in this environment. Doing so requires the user's
Nix devshell (or an equivalent with Boost/ICU/OpenSSL/zstd).

**Resolution / how I worked around it:**
1. Documented the blocker (here) and surfaced it to the user for a decision on
   the build environment.
2. **De-risked the most novel, highest-uncertainty part** — the vector core —
   with a standalone proof-of-concept that depends only on the STL and the
   header-only usearch library, compiled and run with system g++. This validates
   the technical choices (usearch, mmap, exact/approx, keying) that the whole
   design rests on, independently of the QLever build.
3. New, self-contained vector files will be written to compile against the
   minimum dependency surface, so as much as possible can be syntax-checked in
   isolation; engine-coupled files will be marked "needs devshell build".

### Done — usearch fetched and pinned

- `git clone --depth 1 https://github.com/unum-cloud/usearch` → version
  **2.25.3** into `build/_deps/usearch-src`. Header-only; with the default
  toggles (`USEARCH_USE_FP16LIB=0`, `USEARCH_USE_SIMSIMD=0`,
  `USEARCH_USE_OPENMP=0`) it needs **no submodules** — pure STL. Confirms the P0
  plan to add it via `FetchContent` like s2geometry/spatialjoin.

### Done — vector-core proof-of-concept (VERIFIED, compiles + runs)

Source: [`poc/vec_poc.cpp`](./poc/vec_poc.cpp) (reproduce with `poc/run.sh`).
Built: `g++ -std=c++17 -O2 -I build/_deps/usearch-src/include …` → ran clean.

Output:
```
[setup]  20000 vectors, dim=128, key[0]=1000 key[N-1]=140993
[hnsw]   built, size=20000
[recall] HNSW recall@10 over 200 queries = 0.859
[exact]  usearch exact == our brute force: YES (top key ours=1861 usearch=1861)
[mmap]   view(save()) search identical to in-memory: YES (mmapped size=20000)
[subset] exact over 2223 candidates -> top-10 all within subset: YES (nearest dist=0.6739)
```

What this proves for the design:
- **Flat contiguous store** keyed by a non-contiguous `uint64` (standing in for
  QLever's `VocabIndex`, deliberately `key != row`) works — the §5.1 storage
  layout is sound.
- **usearch HNSW** builds and searches; **`save()` then `make(path, view=true)`
  (memory-mapped)** returns *identical* results — confirms the §6 "huge index,
  mmap from disk" serving model.
- **Our exact brute-force kNN matches usearch's built-in exact search exactly** —
  the §8 `ExactAlgorithm` is correct and is a valid ground truth.
- **Exact over a candidate subset** returns only in-subset entities — the §9
  "5 000 green statues → exact over the small set" optimisation is mechanically
  sound.
- usearch keys are 64-bit; `VocabIndex` (< 2⁶⁰) fits directly — no id↔row side
  table needed for the ANN path (§6).

Caveats noted honestly:
- recall@10 = 0.859 with modest params (`connectivity=16`, `expansion_add=128`,
  `expansion_search=64`). It rises toward 1.0 with larger `expansion_search`;
  the PoC value is not a tuned result, just a sanity check that ANN ≈ exact.
- The PoC's "flat store" is an in-RAM `std::vector`; the real `VectorIndex` will
  back it with `ad_utility::MmapVectorView` (same memory layout, paged from
  disk). usearch's own `view()` mmap is already exercised.

### usearch API notes (for the real `VectorIndex`)

- `index_dense_t::make(metric, cfg)` returns a `state_result_t{ index, error }`,
  **not** the index directly — move out `result.index`; check `bool(result)`.
- `index_dense_t::make(path, /*view=*/true)` loads/mmaps from disk (same result
  type).
- `index.reserve(n)` returns `void` (not bool).
- `add(key, const float*)`, `search(const float*, k, thread, /*exact=*/bool)`.
  The `exact` flag gives usearch's own brute force; we keep our own for the
  candidate-subset path.
- Results: `search_result_t::dump_to(Key* keys, distance_t* dists)` → count;
  `distance_t == index_dense_t::distance_t`.
- Metric: `metric_punned_t(D, metric_kind_t::cos_k, scalar_kind_t::f32_k)`.

### Issue #2 — bootstrapping a buildable toolchain without root (RESOLVED)

User asked "can't you build on Docker?" — Docker is installed but **not usable**:
the socket gives `permission denied`, the user is not in the `docker` group, and
there is no `sudo`. The Dockerfile is still informative: it shows QLever builds on
**plain Ubuntu 24.04 with apt packages** (`libboost1.83-dev`, `libicu-dev`,
`libssl-dev`, `libzstd-dev`, …) and the **default gcc-13** — i.e. no Nix, no
Conan needed in principle. But I also can't `apt install` (no root).

**Resolution — userspace Conan:** Conan installs prebuilt dependency binaries into
`~/.conan2` with **no root**. Steps:
1. `python3 -m venv ~/.venvs/qlever` (system pip is PEP-668 "externally managed");
   `pip install "conan>=2" ninja`. → Conan 2.29.1, ninja 1.13.
2. `conan profile detect` (gcc 13, libstdc++11, x86_64).
3. `conan install . -of build-conan --build=missing -s build_type=Release
   -s compiler.cppstd=20` → built Boost/ICU/OpenSSL/zstd from source, generated
   `conan_toolchain.cmake` + a `conan-release` preset.
4. `cmake -S . -B build-conan -G Ninja
   -DCMAKE_TOOLCHAIN_FILE=build-conan/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
   -DUSE_PARALLEL=true -D_NO_TIMING_TESTS=ON -DCOMPILER_SUPPORTS_MARCH_NATIVE=FALSE`.

This gives a real, root-free build using system gcc-13 (the supported compiler).

### Issue #3 — stale FetchContent dependency sources (RESOLVED)

To save re-cloning, I pointed `FETCHCONTENT_SOURCE_DIR_*` at the already-present
`build/_deps/*-src`. The baseline build then failed at 556/734 with two errors:
- `nlohmann/detail/meta/type_traits.hpp: 'EOF' was not declared` (in httpParser).
- `spatialjoin Geo.h: no matching function for dist(Polygon, Point)` (in
  `rdfTypes/GeometryInfo.cpp`, via QLever's `MetricDistanceVisitor` which
  `std::visit`s all 7×7 WKT type pairs).

**Root cause:** those reused sources were the *wrong versions* — left over from a
previous build. `build/_deps/spatialjoin-src` was commit `7f5c900` but
`CMakeLists.txt` pins `c358e479`; reused nlohmann was 3.11.3 but the pin is
3.12.0. The pinned versions compile fine on gcc-13 (the official Docker build
proves gcc-13 works). I had simply forced stale code in.

**Resolution:** verified each reused source's commit against its pin; **s2,
abseil, ctre, antlr, re2, fsst, googletest matched** (kept, to save rebuilds);
dropped the `spatialjoin` and `nlohmann-json` overrides so CMake fetches the
correct pinned versions. Lesson: never reuse a `_deps` tree across checkouts
without verifying commits against the pins.

### Done — Phase P0: usearch added to the build

Added a `FetchContent_Declare(usearch …)` pinned to v2.25.3 (`9fd6b01`) with
`GIT_SUBMODULES ""` (fp16/simsimd unused), a header-only `FetchContent_Populate`
+ `include_directories(SYSTEM …/usearch/include)`, and global
`USEARCH_USE_FP16LIB=0 USEARCH_USE_SIMSIMD=0 USEARCH_USE_OPENMP=0`
(`CMakeLists.txt`). Reconfigure succeeded.

### Done — Issue #4: spatialjoin `bzlib.h` not found (RESOLVED)

With the correct pinned spatialjoin, the build then failed compiling `pb_util`
with `fatal error: bzlib.h`. Cause: spatialjoin's *own* CMake runs
`find_package(BZip2)`; under Conan this **succeeds** (Conan provides bzip2
transitively via Boost.iostreams, which itself `REQUIRES` BZip2), so spatialjoin
defines `PBUTIL_BZIP2_FOUND` and `#include <bzlib.h>` — but it never adds the
include dir to the target, and Conan's bzip2 headers live under `~/.conan2/...`.
In Docker this never triggers (no bzip2 at all → the include is skipped).
- First attempt `-DCMAKE_DISABLE_FIND_PACKAGE_BZip2=TRUE` failed: Boost
  `find_dependency(BZip2 REQUIRED)`, so it can't be disabled.
- **Resolution:** add the Conan bzip2 + zlib include dirs to `CMAKE_CXX_FLAGS`
  (`-isystem …/bzip…/include -isystem …/zlib…/include`) at configure time. A
  build-environment workaround only — **no committed code change**.

### Done — BASELINE BUILD GREEN

`ninja qlever-server qlever-index` → 730/730, **0 failures**. `qlever-index
--help` runs. We now have a real, reproducible, **root-free** QLever build
(userspace Conan + system gcc-13 + usearch). This is the verify loop the rest of
the work depends on.

### Done — Phase P1 (storage core): VERIFIED (compiled + 5/5 tests pass)

New module `src/index/vectorIndex/` (its own `vectorIndex` static lib, linked
into `index`):
- `VectorIndexFormat.h` — enums (`VectorMetric`, `VectorScalar`),
  `VectorIndexConfig`/`VectorIndexMetadata` (+ JSON), `VECTOR_INDEX_VERSION`,
  file-path helpers (`B.vec.N.{meta,keys,data,hnsw}`).
- `VectorIndexBuilder.{h,cpp}` — accumulate `(entity, vector)`, sort by entity
  id, write the mmap flat store (`MmapVector`), optionally build + `save()` the
  usearch HNSW.
- `VectorIndex.{h,cpp}` — pimpl reader (usearch/`MmapVector` kept out of the
  header). `MmapVectorView` keys/data with `AccessPattern::Random`; `getVector`
  (binary search), `searchExact` (over all OR a candidate subset),
  `searchHnsw` (`view()`-mmapped). Exact and HNSW share usearch's
  `metric_punned_t` so distances are identical.
- `test/VectorIndexTest.cpp` — `addLinkAndDiscoverTest(VectorIndexTest
  vectorIndex)`. Verifies: build/open/metadata; `getVector` round-trip + unknown
  entity → nullopt; exact nearest-of-self ≈ distance 0 and ascending order;
  exact restricted to a candidate subset stays in-subset; HNSW top-1 == self and
  recall@10 vs exact ≥ 0.9. **All 5 pass.**

Notes / deviations (honest):
- **Parquet ingest deferred.** Apache Arrow is *not* in the Conan deps and is
  heavy; rather than block the whole pipeline on it, the builder takes
  `(entity, vector)` rows directly. The `VectorInputReader` (Parquet via Arrow)
  is the next step; the builder API is already reader-agnostic.
- **In-memory build.** `VectorIndexBuilder` accumulates vectors in RAM before
  sorting/writing. Fine for now (and the machine is large); external-memory
  build is a later optimisation if needed.

### Done — P1 input reader: `.npy` (VERIFIED, test green)

`VectorInputReader.{h,cpp}`: abstract reader + `NpyVectorInputReader` reading a
2-D little-endian float32 `.npy` matrix + a sidecar IRI-list text file (line `i`
labels row `i`). Dependency-free, ML-native (`np.save` + write IRIs). Parquet via
Apache Arrow remains the planned production reader (interface is format-agnostic).
`test/VectorIndexTest` now 6/6 (added `npyInputReader`, which writes a real v1.0
`.npy` in-test and reads it back).

User decision: "simple reader now, Parquet later" — confirmed this approach.

### Refinement — key the store by `Id` (ValueId), not `VocabIndex`

While checking how to resolve input IRIs to ids I found the canonical path is
`TripleComponent::toValueId(const IndexImpl&) -> optional<Id>` (handles encoded
IRIs via `EncodedIriManager` *and* the vocabulary, returns nullopt if absent).
That returns a full **`Id`/`ValueId`** — which may be a vocab index, an *encoded*
IRI, a blank node, etc. — and is exactly what a join variable binds to at query
time. So the store is now keyed by the raw 64-bit `ValueId` bits (general +
zero-conversion at query time), not `VocabIndex`. The usearch index is
instantiated as `index_dense_gt<std::uint64_t>` (the default key is *signed*,
which would mishandle `ValueId`s whose datatype bits set the high bit; uint64's
only reserved value is `UINT64_MAX`, which is never a valid `ValueId`). Builder,
reader, and tests updated; 6/6 still pass.

### New requirement — query/index-time embedding generation (designed)

User asked for searching by raw text / DB literal / image (url/file/base64), with
QLever generating the embedding. Designed in
[`embedding-generation.md`](./embedding-generation.md). Recommendation: an
`EmbeddingProvider` abstraction over the **OpenAI-compatible `/v1/embeddings`
API** (covers OpenAI, vLLM, TEI, llama.cpp-server, Ollama) over **TCP + Unix
socket** (reuse the Beast HTTP client); optional in-process **llama.cpp** behind
`QLEVER_WITH_LLAMACPP`; **embedding endpoint/model bound to each index's
metadata** (query model must equal index model). Separate track; plugs into the
query path + build pass.

---

## Status & how to continue

**Done + verified (compiled in the real build, 6/6 tests pass):**
- Build environment (root-free Conan + gcc-13) — reproducible; recipe above.
- P0: usearch in the build.
- P1 storage: `VectorIndexFormat`, `VectorIndexBuilder`, `VectorIndex`
  (mmap store + exact + subset + HNSW), `NpyVectorInputReader`.

**Verify loop:** `source ~/.venvs/qlever/bin/activate` then the configure command
in §"Issue #2/#4" (with the bzip2/zlib `-isystem` flags), then
`ninja -C build-conan VectorIndexTest && ./build-conan/test/VectorIndexTest`.

### Done — build pass wired into `qlever index` (VERIFIED end-to-end)

`qlever index` now accepts `--vector-index '<json array>'` and builds vector
indices after the KG index. Implementation:
- `VectorIndexBuildSpec` (config + `.npy`/`.iris` paths) in `VectorIndexFormat.h`;
  `IndexBuilderConfig::vectorIndices_` (`Qlever.h`).
- `--vector-index` option + `parseVectorIndexJson` in `IndexBuilderMain.cpp`.
- `Qlever::buildIndex`: after the KG (and text/materialized-view) build, if
  vector indices are requested, load the index with `doNotLoadPermutations_=true`
  (only the vocab is needed) and run `buildVectorIndicesPass` (`Qlever.cpp`),
  which resolves each input IRI via `TripleComponent::toValueId(index.getImpl())`
  and feeds `VectorIndexBuilder`; unknown IRIs are skipped + counted.

End-to-end test: tiny `kg.ttl` (5 entities) + `vectors.npy` (6×4, one row an
unknown IRI) → `qlever index -i kgindex -f kg.ttl -F ttl --vector-index '...'`
logged "indexed 5 vectors, skipped 1 (IRI not in the knowledge graph), HNSW=yes"
and produced `kgindex.vec.clip.{meta,keys,data,hnsw}` with correct metadata.

### Issue #5 — `U_FILE_ACCESS_ERROR` at index-build runtime (RESOLVED)

`qlever-index` failed with ICU's `U_FILE_ACCESS_ERROR` (collation init couldn't
find ICU data). Conan ships the data as `…/icu…/p/res/icudt76l.dat` but doesn't
auto-set the env var in this setup. **Resolution:** export
`ICU_DATA=$(dirname <icudt*.dat>)` before running the binaries (a runtime env
detail; the conan `conanrun.sh` can also be sourced). Add this to the verify
loop. (Unit tests don't hit it — they don't build a vocab.)

### Done — load wiring (VERIFIED: server loads at startup)

- `IndexImpl` holds `HashMap<string, VectorIndex> vectorIndices_` +
  `getVectorIndex(name)` / `vectorIndices()` accessors and `loadVectorIndices()`,
  which **auto-detects** `<base>.vec.<name>.meta` files (no manifest change, no
  flag) and memory-maps each. Called from `createFromOnDiskIndex` after the vocab
  loads, so both the server and the embedded `Qlever` get it for free.
- `Index::getVectorIndex(name)` delegates to the impl (forward-declared
  `qlever::vector::VectorIndex` in `Index.h` to avoid a heavy include). This is
  how query operations will reach the index:
  `getExecutionContext()->getIndex().getVectorIndex(name)`.

Verified: `qlever-server -i kgindex` logs
"Loaded vector index 'clip' (5 vectors, dim 4, HNSW=yes)" and serves. (Needs
`ICU_DATA` set, as for the builder.)

### Done — query path (leaf `SERVICE vec:`) VERIFIED end-to-end in SPARQL

Implemented a standalone vector-search SERVICE that produces a `(?result,
?score)` top-k table (the query point is an explicit vector or a constant
entity's vector). Files:
- `VECTOR_SEARCH_IRI` (`MagicServiceIriConstants.h`).
- `engine/VectorSearchConfig.h` — dependency-light `VectorSearchConfiguration`.
- `parser/VectorSearchQuery.{h,cpp}` — `MagicServiceQuery` subclass; parses
  `vec:index`, `vec:query <iri>`, `vec:queryVector "f,f,…"`, `vec:result`,
  `vec:bindScore`, `vec:k`, `vec:algorithm`; lowers to the config.
- `engine/VectorSearch.{h,cpp}` — leaf `Operation` (no children); resolves the
  query vector (explicit or via `TripleComponent::toValueId` + `getVector`),
  runs HNSW or exact search, emits an `IdTable` of `(entity Id, Double score)`.
- Wiring: variant alternative (`GraphPatternOperation.h`), parser dispatch
  (`SparqlQleverVisitor.cpp`), planner leaf branch in
  `graphPatternOperationVisitor` (`QueryPlanner.cpp`). No join-hook needed (leaf).

**Issue #6 — variant exhaustiveness static_asserts (RESOLVED).** Adding a
`GraphPatternOperation` variant alternative broke two compile-time enumerations:
`CheckUsePatternTrick.cpp` (`SameAsAny<…magic services…>`) and
`GraphPatternAnalysis.h` (`SimilarToAny<…all graph-pattern ops…>`). Added
`VectorSearchQuery` to both. (This is exactly the per-service core-file churn that
the planned **Phase R** registry refactor will eliminate — noted as motivation.)

End-to-end test against `kgindex` (entities e0..e4 with unit/axis vectors):
```
# vec:queryVector "1,0,0,0" k=3  ->  e0=0.0, e4=0.293, e3=1.0   (cosine distances exact)
# vec:query <http://ex/e0> k=2   ->  e0=0.0, e4=0.293           (entity-vector lookup works)
```
Both modes correct; results exported as IRIs + scores. `VectorIndexTest` still 6/6.

### Done — binary-join "for each ?x" form VERIFIED end-to-end

`engine/VectorSearchJoin.{h,cpp}`: a single-child operation whose child is the
nested query pattern (binds the `<left>` variable). For each child row it looks up
the query entity's vector and emits the k nearest index entities, carrying the
child's columns through and adding `?result`/`?right` (+ optional `?score`).
Config gained `leftVariable_`; the parser handles `<left>` and `<right>` and
validates leaf-vs-binary; the planner's new `visitVectorSearch` dispatches: leaf
→ `VectorSearch`, binary → plan the nested pattern + construct `VectorSearchJoin`
(no `createJoinCandidates` hook needed for the nested-child design).

End-to-end (kgindex e0..e4):
```
SERVICE vec: { { ?statue <p> ?l . }
  _:c vec:index "clip" ; vec:left ?statue ; vec:right ?similar ; vec:k 2 ; vec:bindScore ?score }
# -> e0:{e0=0.0,e4=0.293}, e1:{e1=0.0,e4=0.293}, e2:{e2=0.0,e4=1.0}, ...  (all exact)
```

### Done — Embedding track E1: search by text VERIFIED end-to-end

- Per-index embedding endpoint in metadata: `VectorIndexConfig.embeddingUrl` +
  `embeddingModel` (persisted in `.meta`, set via the `--vector-index` JSON keys
  `embeddingUrl`/`embeddingModel`). Binds the query model to the index (the
  same-model invariant).
- `engine/EmbeddingClient.{h,cpp}`: `embedTextOpenAI(baseUrl, model, text,
  handle)` — POSTs `{"model","input":[text]}` to `<baseUrl>/v1/embeddings`
  (OpenAI-compatible; works with vLLM/TEI/llama.cpp-server/Ollama) via QLever's
  Boost.Beast `sendHttpOrHttpsRequest`, parses `data[0].embedding`.
- `vec:queryText "…"` parsed (leaf form, mutually exclusive with the other query
  specs); `VectorSearch::computeResult` embeds it via the index's endpoint, then
  searches.
- Test harness: `build-conan/vectest/mock_embed.py` (a ~30-line OpenAI-compatible
  `/v1/embeddings` stub returning deterministic 4-d vectors).

End-to-end: rebuilt `kgindex` with `embeddingUrl=http://127.0.0.1:8088`, started
the mock + server, ran `SERVICE vec: { _:c vec:index "clip" ; vec:queryText
"axis2" ; vec:result ?sim ; vec:k 2 ; vec:bindScore ?score }` → `e2`=0.0,
`e4`=1.0 (the mock maps "axis2" → [0,0,1,0]; nearest is e2). The full path
text → HTTP → embedding → search → IRIs+scores is verified.

### Done — Embedding E2a: Unix-socket transport VERIFIED

`EmbeddingClient` now dispatches on the endpoint URL: `unix:/path.sock` (or
`unix:///path.sock`) → a synchronous Beast request over a
`boost::asio::local::stream_protocol` socket (the recommended production path: a
local vLLM/llama.cpp server on a Unix socket, off the network); `http(s)://` →
the existing shared HTTP client. Verified end-to-end with an AF_UNIX mock
(`build-conan/vectest/mock_embed_unix.py`): `vec:queryText "axis3"` over a `unix:`
endpoint → `[0,0,0,1]` → nearest entity e3.

Image inputs (`vec:queryImage`, url/file/base64) are **E2b**, deferred: the
OpenAI embeddings spec is text-only and multimodal/image servers use varying
request shapes, so the concrete request format should be fixed against a specific
target endpoint (e.g. a vLLM CLIP server) before implementing — the plumbing
(file/url/base64 → bytes → POST) is otherwise identical to the text path.

### Done — Embedding E3: build a vector index from text VERIFIED

`qlever index --vector-index` now accepts `"texts"` instead of `"npy"`: a
row-aligned file of texts that are embedded at index time via the index's
`embeddingUrl` (each `(iri, text)` -> resolve iri, embed text, add). Dimension is
inferred from the first embedding. `Qlever.cpp` gained `buildFromTexts`
(alongside `buildFromNpy`); `IndexBuilderMain` parses `texts` and enforces
exactly one of `npy`/`texts`.

Verified end-to-end: built `kgindex`'s `clip` index from `texts.txt`
(axis0..axis3) with the mock embedder running, then `vec:queryText "axis2"`
returned e2. The full loop now needs NO precomputed vectors: text in -> indexed
-> queryable by text.

### Done — Phase R (R-minimal): magic-service registry, VERIFIED

Made magic `SERVICE`s extensible so a new one needs no edits to the core
parser/planner dispatch or exhaustiveness checks:
- `parsedQuery::MagicService { shared_ptr<MagicServiceQuery> }` — one generic
  variant node (replaces per-service alternatives).
- `parser/MagicServiceRegistry` — IRI → factory; services self-register (a static
  initializer in `VectorSearchQuery.cpp`). The parser dispatch
  (`SparqlQleverVisitor`) gained ONE generic registry branch and lost the
  per-service `else if`.
- Planner: `graphPatternOperationVisitor` gained ONE `MagicService` branch →
  `GraphPatternPlanner::planMagicService`, which dispatches via a
  `type_index → handler` registry. Vector's handler is one entry there.
- Vector removed from: the variant, the parser if/else, the planner
  if-constexpr, and both static_assert enumerations
  (`CheckUsePatternTrick`, `GraphPatternAnalysis`).

**Result:** adding a magic service now = a `MagicServiceQuery` subclass (self-
registers its parser) + one handler entry in `planMagicService`. The 6 former
seams are down to 1 in-tree registration point.

**Residual (documented):** the planner handler must live in `QueryPlanner.cpp`
because `GraphPatternPlanner` is a private nested type and `makeSubtreePlan` is
file-local — so fully external (out-of-tree) planner handlers would need those
exposed. The parser side is fully self-registering. Existing services
(spatial/text/path/…) were intentionally left on their dedicated alternatives
(zero risk); they can be migrated incrementally.

**Verified:** vector queries (queryVector / for-each) work through the registry;
a normal SPARQL query still plans; **regression: SparqlParserTest 16/16,
QueryPlannerTest 107/107, VectorIndexTest 6/6**.

### Done — Restricted-right exact-subset optimization VERIFIED

The original "5000 green statues → exact" optimisation. The query-point forms
(`vec:queryVector`/`vec:query`/`vec:queryText`) now accept an OPTIONAL nested
`{ ... }` pattern that binds `?result`; when present, the search is restricted to
exactly those candidate entities via `VectorIndex::searchExact(query, k,
candidates)` (correct, and faster than filtering the whole HNSW for a selective
set). `VectorSearch` gained an optional `candidates_` child; the planner plans the
nested pattern and passes it; candidate ids are de-duplicated.

Verified: query `[1,0,0,0]` restricted to `{e1,e4}` → `e4=0.0, e1=1.0` (e0, the
*global* nearest, is correctly excluded); unrestricted → `e0,e4`.

### Done — Embedding E2b: image inputs VERIFIED

`vec:imageUrl <iri|"url">`, `vec:imageFile "path"`, `vec:imageBase64 "b64"` are
parsed into a `VectorSearchConfiguration::ImageQuery`. At query time the op builds
the payload (URL passed through; file read + `absl::Base64Escape` → data URI;
base64 wrapped into a data URI) and embeds it via `embedImageOpenAI`
(EmbeddingClient refactored to share the POST/parse helper between text & image,
keeping the TCP + unix-socket dispatch).

Verified end-to-end against the mock: `imageUrl <http://x/axis2.png>` → e2;
`imageFile` (content "axis1") → e1; `imageBase64`("axis3") → e3.

Caveat: multimodal embedding endpoints vary in their exact request shape; this
uses the common "image payload as the `input` string" convention. The exact
field can be made configurable per endpoint later.

### Done — Embedding E4: optional in-process llama.cpp backend (scaffold)

- CMake `option(QLEVER_WITH_LLAMACPP OFF)`; when ON, FetchContent llama.cpp
  (pinned b4991), define `QLEVER_WITH_LLAMACPP`, link `llama` into `engine`.
- `EmbeddingClient`: a `llama:/path/model.gguf` endpoint routes to an in-process
  `embedViaLlamaCpp` (loads the GGUF once per path, mean-pooled + L2-normalized
  embeddings) guarded by `#ifdef QLEVER_WITH_LLAMACPP`. When the flag is off, the
  `llama:` scheme throws a clear "rebuild with -DQLEVER_WITH_LLAMACPP=ON" error.

Verification: the **default-OFF build is clean** (the guard excludes the code);
and the guarded llama.cpp code **compiles cleanly against the real b4991
`llama.h`** (syntax-checked with the flag on + the fetched headers). NOT verified
here: actual in-process inference (needs a full llama.cpp build + a GGUF model,
infeasible in this environment) and the heavy FetchContent build itself.

### Done — Plugin architecture: services in their own folder, zero core edits

Turned magic SERVICEs into drop-in plugins so a custom service (and our fork) is
just a folder, rebased on master with no core conflicts.
- `engine/MagicServicePlanning.{h,cpp}`: a stable `MagicServicePlanningContext`
  façade (`qec()`, `addLeafOperation`, `addOperationWithChildPattern` — hides
  `GraphPatternPlanner`/`makeSubtreePlan`/`SubtreePlan`) + a
  `MagicServicePlannerRegistry`. `QueryPlanner::planMagicService` is now fully
  generic; `visitVectorSearch` and all vector includes were removed from the
  planner. **No per-service code in QueryPlanner anymore.**
- `src/services/` with a glob-based CMake that builds each service folder as an
  OBJECT library (self-registration survives linking) and links them into the
  binaries; one-time `add_subdirectory(src/services)` in the top-level CMake.
- Moved the vector QUERY code to `src/services/vectorSearch/`
  (`VectorSearch*.{h,cpp}`, `VectorSearchConfig.h`, `VectorSearchQuery.{h,cpp}`,
  and `VectorSearchService.cpp` which registers BOTH the parser factory and the
  planner handler). The IRI constant moved out of core too. `EmbeddingClient`
  and the on-disk storage/build/load stayed in core (shared / index-lifecycle).

Issues hit + fixed: `target_link_libraries` plain-vs-keyword signature clash on
the binaries (used the plain form); the moved `VectorSearchQuery.cpp` needed an
explicit `#include "parser/SparqlTriple.h"` (previously transitive).

Verified: all vector query forms work via the registries (object-lib self-reg
survived); regression SparqlParserTest 16/16, QueryPlannerTest 107/107,
VectorIndexTest 6/6.

### Done — Index-extension registries: vector storage fully decoupled

Generalized the on-disk-index coupling so a service with its own index is also
fully in its folder.
- `index/IndexExtension.{h,cpp}`: `IndexExtensionRegistry` with **build hooks**
  `(const Index&, basename, json)` run by `qlever index` after the main index is
  built (URI→Id available) and **load hooks** `(IndexImpl&, basename)` run at
  server start after the vocab. `IndexImpl` got a generic
  `extensions_` store + `setExtension`/`getExtension` (replacing the
  vector-specific `vectorIndices_`/`getVectorIndex`/`loadVectorIndices`).
- Core builder: one generic `--service-index '<json>'` option (object keyed by
  service name) → `IndexBuilderConfig::serviceIndexJson_`; `Qlever::buildIndex`
  loads the index and calls every registered build hook (no vector code).
- Moved into `src/services/vectorSearch/`: `VectorIndex`/`VectorIndexBuilder`/
  `VectorInputReader`/`VectorIndexFormat`, `EmbeddingClient`, and a new
  `VectorIndexExtension.{h,cpp}` (the `VectorIndexCollection`, a `getVectorIndex`
  helper reading `getExtension`, and the build+load hook registrations). CMake:
  a small `vectorStorage` static lib (also used by the storage test) + the
  service OBJECT lib; the `http` and optional `llama` links moved here too.

**Core now has zero vector references** (verified by grep). Issues fixed: the
storage unit test relinked against `vectorStorage` (was `vectorIndex`);
`EmbeddingClient` uses `http`, exposed PUBLIC on the object lib so it propagates
to the binaries.

Verified end-to-end via the generic path: `qlever index --service-index
'{"vectorSearch":[…]}'` builds (build hook); server loads (load hook); all query
forms correct. Regression SparqlParserTest 16/16, QueryPlannerTest 107/107,
VectorIndexTest 6/6.

**To add a new SERVICE now:** create `src/services/<name>/` with a
`MagicServiceQuery` subclass, your `Operation`(s), a `<Name>Service.cpp` that
registers the parser factory + planner handler, and a tiny `CMakeLists.txt`
(OBJECT lib + `set_property(GLOBAL APPEND PROPERTY QLEVER_SERVICE_OBJLIBS …)`).
Re-run cmake. Zero edits to core engine/parser/variant/CMake.

**Remaining (explicitly deferred by the user): R-full
(migrate other services onto the registry) and Parquet ingest. Other possible
future work: batch-embed at index time over a SPARQL column; quantised stores
(f16/i8/b1) end-to-end.
2. Phase R (magic-service registry) to stop editing core files per service, and
   to add the outside-left + restricted-right-subset (exact-over-candidates)
   form via a shared `IncompleteBinaryJoin` planner hook.
3. Embedding E3/E4: batch indexing from a text/image column; optional in-process
   llama.cpp backend.
1b. (legacy note) resolve input IRIs → `VocabIndex` against the on-disk
1b. (legacy note) resolve input IRIs → `VocabIndex` against the on-disk
   vocab (mirror `TextIndexBuilder`'s vocab reload), then `VectorIndexBuilder`.
   Either a small `qlever-vector-index` tool (least invasive first step) or fold
   into `qlever index` (`IndexBuilderConfig` + `Qlever::buildIndex`). NOTE: IRI→Id
   resolution must use the vocab's stored form / `EncodedIriManager` — verify the
   exact lookup before relying on it.
2. **Load wiring**: `IndexImpl` registry of `VectorIndex` + `loadVectorIndices`
   (mirror `addTextFromOnDiskIndex`), `Index` accessor, call from `Server` +
   `Qlever` ctor.
3. **Query path**: `VectorSearchQuery` (parser), `VectorSearchJoin` (operation) +
   `VectorSearchAlgorithms`, planner wiring — preceded by **Phase R** (magic-
   service registry refactor) so no core files are edited per service.
4. **Embedding track (E1–E4)**: per `embedding-generation.md`.

Work is on branch `feat/vector-index`, not committed (awaiting request).
