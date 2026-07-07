// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// End-to-end test of the core "index-payload" vector-search flow (see
// `docs/vector-index/index-payload-design.md`): vectors enter QLever as a
// `.npy` bundle (matrix + row-aligned IRI sidecar) through the REGISTERED
// build hook, are loaded through the REGISTERED load hook, and are reached
// from SPARQL only via the `vec:distance` function -- no vector ever becomes
// an RDF term. Ranking is plain `BIND` + `FILTER(BOUND)` + `ORDER BY`;
// entities without a vector get an `UNDEF` distance.

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "./util/GTestHelpers.h"
#include "./util/IndexTestHelpers.h"
#include "engine/QueryPlanner.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "parser/SparqlParser.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/json.h"

namespace {
using ad_utility::testing::getQec;
using ad_utility::testing::makeGetId;

constexpr std::string_view PREFIX =
    "PREFIX vec: <https://qlever.cs.uni-freiburg.de/vectorSearch/>\n"
    "PREFIX vidx: <https://qlever.cs.uni-freiburg.de/vectorSearch/index/>\n";

// The test knowledge graph: four entities that get a vector below, plus
// `<nomember>`, which is in the graph but deliberately NOT in the vector
// index. `<is-a> <Item>` is the "natural enumerator" pattern of the design
// doc: the real predicate the embeddings were derived from.
constexpr std::string_view KG =
    "<a> <is-a> <Item> . <b> <is-a> <Item> . <c> <is-a> <Item> . "
    "<d> <is-a> <Item> . <nomember> <is-a> <Item> .";

// The dimension-3 vectors, row i of the `.npy` matrix belonging to line i of
// the IRI sidecar. The sidecar deliberately MIXES bare IRIs and `<...>`
// irirefs -- both forms are accepted, so every test on this fixture also
// asserts that bare-IRI rows resolve. With the cosine metric and the query
// point [1,0,0]: `<a>` and `<c>` are exact matches (distance ~0), `<b>` is at
// 1-0.6 = 0.4, and `<d>` is orthogonal (distance 1) -- an unambiguous
// ranking.
const std::vector<std::pair<std::string, std::vector<float>>>& testVectors() {
  static const std::vector<std::pair<std::string, std::vector<float>>> vecs{
      {"a", {1.f, 0.f, 0.f}},
      {"<b>", {0.6f, 0.8f, 0.f}},
      {"c", {1.f, 0.f, 0.f}},
      {"<d>", {0.f, 0.f, 1.f}},
  };
  return vecs;
}

// Encode `value` as the little-endian bf16 bit pattern that
// `ml_dtypes.bfloat16` puts into a `.npy` file: the top 16 bits of the fp32.
// Exact for the bf16-representable test values (0, 0.25, 0.5, 1) --
// truncation only ever drops zero bits there.
std::array<char, 2> bf16LittleEndian(float value) {
  uint32_t asUint = 0;
  std::memcpy(&asUint, &value, sizeof(asUint));
  const auto bits = static_cast<uint16_t>(asUint >> 16);
  return {static_cast<char>(bits & 0xff), static_cast<char>(bits >> 8)};
}

// Write `rows` as a NumPy v1.0 `.npy` file (C-order; little-endian float32
// `<f4`, or -- with `bf16` -- the 2-byte `<V2` void dtype that
// `ml_dtypes.bfloat16` serializes as) plus the row-aligned IRI sidecar -- the
// exact input bundle of the design.
void writeNpyBundle(
    const std::string& npyPath, const std::string& irisPath,
    const std::vector<std::pair<std::string, std::vector<float>>>& rows,
    bool bf16 = false) {
  const size_t numRows = rows.size();
  const size_t dim = rows.front().second.size();
  std::string dict = std::string{"{'descr': '"} + (bf16 ? "<V2" : "<f4") +
                     "', 'fortran_order': False, 'shape': (" +
                     std::to_string(numRows) + ", " + std::to_string(dim) +
                     "), }";
  size_t pad = (64 - ((10 + dict.size() + 1) % 64)) % 64;
  dict.append(pad, ' ');
  dict.push_back('\n');
  std::ofstream out{npyPath, std::ios::binary};
  out.write("\x93NUMPY", 6);
  char version[2] = {1, 0};
  out.write(version, 2);
  uint16_t headerLen = static_cast<uint16_t>(dict.size());
  char lenBytes[2] = {static_cast<char>(headerLen & 0xff),
                      static_cast<char>((headerLen >> 8) & 0xff)};
  out.write(lenBytes, 2);
  out.write(dict.data(), dict.size());
  std::ofstream irisOut{irisPath};
  for (const auto& [iri, vec] : rows) {
    if (bf16) {
      for (float value : vec) {
        auto bytes = bf16LittleEndian(value);
        out.write(bytes.data(), bytes.size());
      }
    } else {
      out.write(reinterpret_cast<const char*>(vec.data()),
                vec.size() * sizeof(float));
    }
    irisOut << iri << "\n";
  }
}

// The shared test context: build the vector index "emb" from the `.npy`
// bundle via the REGISTERED build hook (the `--service-index` path:
// `parseSpec` -> `buildFromNpy` -> IRI resolution against the KG vocabulary),
// then attach it via the REGISTERED load hook -- the same two hooks `qlever
// index` and the server run. Idempotent (the `getQec` context is cached).
QueryExecutionContext* qecWithPayloadIndex() {
  QueryExecutionContext* qec = getQec(std::string{KG});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, testVectors());
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"emb","npy":")" + npy + R"(","iris":")" +
      iris + R"(","metric":"cosine","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  // The load hook auto-materialized the per-index metadata triples as DELTA
  // triples, but the cached `getQec` context captured its located-triples
  // snapshot at construction -- refresh it so queries see them (the server
  // takes a fresh snapshot per query; the test context does not).
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  // The load hook memory-maps the index, which keeps the data alive; the
  // directory entries (and the input bundle) can go right away.
  for (auto* suffix :
       {".vec.emb.meta", ".vec.emb.keys", ".vec.emb.rowmap", ".vec.emb.data",
        ".vec.emb.iris", ".vec.emb.hnsw", ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
  }
  return qec;
}

// Parse, plan, and return the execution tree of `query`.
QueryExecutionTree planQuery(QueryExecutionContext* qec, std::string query) {
  ParsedQuery pq = SparqlParser::parseQuery(
      &qec->getIndex().getImpl().encodedIriManager(), std::move(query));
  auto handle = std::make_shared<ad_utility::CancellationHandle<>>();
  QueryPlanner planner{qec, handle};
  return planner.createExecutionTree(pq);
}

// ===========================================================================
// The bf16-STORAGE fixture: a SEPARATE knowledge graph (a distinct `getQec`
// cache entry) whose vector index is built with `"scalar": "bf16"` from an
// fp32 `.npy` bundle (the fp32-input path; the native `<V2` bf16 INPUT is the
// fixture further below). Every vector component is exactly
// bf16-representable (0, 0.25, 0.5, 1), so truncating the fp32 input to the
// 2-byte bf16 store is lossless and the cosine distances are exact.
constexpr std::string_view KG_BF16 =
    "<a> <is-a> <Bf16Item> . <b> <is-a> <Bf16Item> . "
    "<c> <is-a> <Bf16Item> . <d> <is-a> <Bf16Item> .";

// The dimension is deliberately large enough that the page-rounded on-disk
// size of the flat store distinguishes 2-byte bf16 from 4-byte fp32 rows
// (tiny matrices land in a single page either way).
constexpr size_t BF16_DIM = 1024;
constexpr size_t BF16_NUM_VECTORS = 4;

// `<a>` and `<c>` equal the basis vector e0 (distance 0 to an e0 query),
// `<b>` is at 45 degrees in the e0/e1 plane (cosine distance 1 - 1/sqrt(2)),
// and `<d>` is orthogonal (distance 1).
std::vector<std::pair<std::string, std::vector<float>>> bf16TestVectors() {
  std::vector<float> e0(BF16_DIM, 0.f);
  e0[0] = 1.f;
  std::vector<float> diag(BF16_DIM, 0.f);
  diag[0] = 0.5f;
  diag[1] = 0.5f;
  std::vector<float> e2(BF16_DIM, 0.f);
  e2[2] = 1.f;
  return {{"<a>", e0}, {"<b>", diag}, {"<c>", e0}, {"<d>", e2}};
}

// Build the "embbf16" index through the registered hooks and return the qec
// together with the on-disk size of its `.vec.embbf16.data` flat store
// (measured before the fixture unlinks the mmap-ed files).
std::pair<QueryExecutionContext*, std::uintmax_t> qecWithBf16Index() {
  static std::uintmax_t dataFileSize = 0;
  QueryExecutionContext* qec = getQec(std::string{KG_BF16});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return {qec, dataFileSize};
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-bf16-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, bf16TestVectors());
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"embbf16","npy":")" + npy + R"(","iris":")" +
      iris + R"(","metric":"cosine","scalar":"bf16","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  dataFileSize = std::filesystem::file_size(basename + ".vec.embbf16.data");
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  for (auto* suffix :
       {".vec.embbf16.meta", ".vec.embbf16.keys", ".vec.embbf16.rowmap",
        ".vec.embbf16.data", ".vec.embbf16.iris", ".vec.embbf16.hnsw",
        ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
  }
  return {qec, dataFileSize};
}

// ===========================================================================
// The NATIVE-bf16-INPUT fixture: the `.npy` matrix itself is bf16 -- dtype
// `<V2`, the opaque 2-byte void that `ml_dtypes.bfloat16` serializes as
// (numpy has no native bf16 type code); each scalar is the little-endian
// bf16 bit pattern. Combined with `"scalar": "bf16"` the values go into the
// 2-byte store with no fp32 round-trip in the file. The sidecar uses BARE
// IRIs throughout. All components (0, 0.25, 0.5, 1) are exactly
// bf16-representable, so the ranking is exact.
constexpr std::string_view KG_NATIVE_BF16 =
    "<a> <is-a> <NativeBf16Item> . <b> <is-a> <NativeBf16Item> . "
    "<c> <is-a> <NativeBf16Item> . <d> <is-a> <NativeBf16Item> .";

// `<a>`/`<c>` = e0 (distance ~0 to an e0 query), `<b>` at 45 degrees in the
// e0/e1 plane (cosine distance 1 - 1/sqrt(2)), `<d>` orthogonal (distance 1).
std::vector<std::pair<std::string, std::vector<float>>> nativeBf16Vectors() {
  return {{"a", {1.f, 0.f, 0.f, 0.f}},
          {"b", {0.25f, 0.25f, 0.f, 0.f}},
          {"c", {1.f, 0.f, 0.f, 0.f}},
          {"d", {0.f, 0.f, 0.5f, 0.f}}};
}

QueryExecutionContext* qecWithNativeBf16Index() {
  QueryExecutionContext* qec = getQec(std::string{KG_NATIVE_BF16});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-nbf16-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, nativeBf16Vectors(), /*bf16=*/true);
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"embnative","npy":")" + npy +
      R"(","iris":")" + iris +
      R"(","metric":"cosine","scalar":"bf16","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  for (auto* suffix :
       {".vec.embnative.meta", ".vec.embnative.keys", ".vec.embnative.rowmap",
        ".vec.embnative.data", ".vec.embnative.iris", ".vec.embnative.hnsw",
        ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
  }
  return qec;
}

// ===========================================================================
// The CROSS-INDEX fixture: one knowledge graph, THREE vector indices (same
// dimension 3, cosine, f32) built in a single spec, with DISTINCT entities:
//  * `idxa` (model "clip"):  <a1>=[1,0,0], <a2>=[0.6,0.8,0], <a3>=[0,0,1]
//                            (<a4> is an AItem WITHOUT a vector);
//  * `idxb` (model "clip"):  <b1>=[1,0,0], <b2>=[0,1,0] -- the SAME embedding
//                            space as `idxa`, so `vec:vector(idxb, ...)` is
//                            comparable in `idxa`;
//  * `idxother` (model "other"): <o1>=[1,0,0] -- same dim/precision but a
//                            DIFFERENT model, so it must NOT be comparable.
constexpr std::string_view KG_CROSS =
    "<a1> <is-a> <AItem> . <a2> <is-a> <AItem> . <a3> <is-a> <AItem> . "
    "<a4> <is-a> <AItem> . <b1> <is-a> <BItem> . <b2> <is-a> <BItem> . "
    "<o1> <is-a> <OItem> .";

// The typed query-vector datatype IRIs used in the queries below (bracketless
// prefix; see `VEC_QUERY_DATATYPE_PREFIX`).
constexpr std::string_view VEC_DT_CLIP_F32 =
    "https://qlever.cs.uni-freiburg.de/vectorSearch/vec/clip/f32";
constexpr std::string_view VEC_DT_CLIP_F16 =
    "https://qlever.cs.uni-freiburg.de/vectorSearch/vec/clip/f16";

QueryExecutionContext* qecWithCrossIndexes() {
  QueryExecutionContext* qec = getQec(std::string{KG_CROSS});
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-cross-" + std::to_string(::getpid())))
          .string();
  writeNpyBundle(basename + ".a.npy", basename + ".a.iris",
                 {{"<a1>", {1.f, 0.f, 0.f}},
                  {"<a2>", {0.6f, 0.8f, 0.f}},
                  {"<a3>", {0.f, 0.f, 1.f}}});
  writeNpyBundle(basename + ".b.npy", basename + ".b.iris",
                 {{"<b1>", {1.f, 0.f, 0.f}}, {"<b2>", {0.f, 1.f, 0.f}}});
  writeNpyBundle(basename + ".o.npy", basename + ".o.iris",
                 {{"<o1>", {1.f, 0.f, 0.f}}});
  auto entry = [&](const std::string& name, const std::string& input) {
    return nlohmann::json{{"name", name},
                          {"npy", basename + "." + input + ".npy"},
                          {"iris", basename + "." + input + ".iris"},
                          {"metric", "cosine"},
                          {"hnsw", false}};
  };
  nlohmann::json spec{
      {"vectorSearch",
       nlohmann::json::array(
           {entry("idxa", "a"), entry("idxb", "b"), entry("idxother", "o")})}};
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  // The embedding model is a serving concern, set at server start rather than
  // baked into the index. Give idxa/idxb model "clip" and idxother model
  // "other" via the endpoints env override so the load hook stamps each
  // index's query-vector datatype (which drives cross-index comparability in
  // the tests below).
  ::setenv(qlever::vector::VECTOR_SEARCH_ENDPOINTS_ENV_VAR,
           R"({"idxa":{"embeddingModel":"clip"},)"
           R"("idxb":{"embeddingModel":"clip"},)"
           R"("idxother":{"embeddingModel":"other"}})",
           /*overwrite=*/1);
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  ::unsetenv(qlever::vector::VECTOR_SEARCH_ENDPOINTS_ENV_VAR);
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  for (std::string_view name : {"idxa", "idxb", "idxother"}) {
    for (std::string_view suffix :
         {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
      std::error_code ec;
      std::filesystem::remove(
          basename + ".vec." + std::string{name} + std::string{suffix}, ec);
    }
  }
  for (std::string_view suffix :
       {".a.npy", ".a.iris", ".b.npy", ".b.iris", ".o.npy", ".o.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + std::string{suffix}, ec);
  }
  return qec;
}

// ===========================================================================
// The LARGE fixture: an index with MORE entities than the `vec:distance`
// parallel-scan threshold (see `VEC_DISTANCE_PARALLEL_THRESHOLD` in
// `VectorDistanceExpression.cpp`), so a whole-column `BIND(vec:distance(...))`
// runs the multi-threaded per-row scan (5000 rows is a single BIND evaluation
// block: below the 10'000 `Bind::CHUNK_SIZE`, above the 2048 parallel
// threshold). Used to prove the parallel path is bit-identical to a serial
// reference and race-free.
constexpr size_t BIG_NUM_VECTORS = 5000;
constexpr size_t BIG_DIM = 8;

// An exactly-representable (dyadic) query vector, so the string the SPARQL
// query carries (`BIG_QUERY_STR`) parses -- via the same `absl::from_chars`
// the expression uses -- to EXACTLY these `float`s. That lets the serial
// reference `makeDistanceComputer(BIG_QUERY)` use a bit-identical query point,
// so any difference the test sees comes from the parallel scan, not from query
// re-encoding.
const std::vector<float> BIG_QUERY = {0.5f,  -0.25f, 0.75f,  1.0f,
                                      -0.5f, 0.25f,  0.125f, -0.125f};
constexpr std::string_view BIG_QUERY_STR =
    "0.5,-0.25,0.75,1,-0.5,0.25,0.125,-0.125";

// Deterministic, distinct, non-zero-norm vectors for `<e0>..<e{N-1}>` (a
// xorshift64 stream mapped to [-1, 1), with a +2 bump on the first component
// so no row is the zero vector -- cosine distance is NaN there).
std::vector<std::pair<std::string, std::vector<float>>> bigTestVectors() {
  std::vector<std::pair<std::string, std::vector<float>>> rows;
  rows.reserve(BIG_NUM_VECTORS);
  uint64_t state = 88172645463325252ull;
  auto nextFloat = [&state]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    uint32_t bits = static_cast<uint32_t>(state >> 40);  // top 24 bits
    return static_cast<float>(bits) / static_cast<float>(1 << 23) - 1.0f;
  };
  for (size_t i = 0; i < BIG_NUM_VECTORS; ++i) {
    std::vector<float> v(BIG_DIM);
    for (size_t j = 0; j < BIG_DIM; ++j) {
      v[j] = nextFloat();
    }
    v[0] += 2.0f;
    rows.emplace_back("<e" + std::to_string(i) + ">", std::move(v));
  }
  return rows;
}

QueryExecutionContext* qecWithLargeIndex() {
  // One triple per line: the parallel Turtle parser batches the input and needs
  // a statement terminator (`.` + newline) within each batch, so a single
  // 150 KB line would trip its buffer.
  std::string kg;
  for (size_t i = 0; i < BIG_NUM_VECTORS; ++i) {
    kg += "<e" + std::to_string(i) + "> <is-a> <BigItem> .\n";
  }
  QueryExecutionContext* qec = getQec(kg);
  auto& impl = const_cast<Index&>(qec->getIndex()).getImpl();
  if (impl.getExtension(std::string{qlever::vector::VECTOR_EXTENSION_NAME}) !=
      nullptr) {
    return qec;
  }
  std::string basename =
      (std::filesystem::temp_directory_path() /
       ("qlever-vecpayloade2e-big-" + std::to_string(::getpid())))
          .string();
  std::string npy = basename + ".input.npy";
  std::string iris = basename + ".input.iris";
  writeNpyBundle(npy, iris, bigTestVectors());
  nlohmann::json spec = nlohmann::json::parse(
      R"({"vectorSearch":[{"name":"embbig","npy":")" + npy + R"(","iris":")" +
      iris + R"(","metric":"cosine","hnsw":false}]})");
  for (const auto& hook : IndexExtensionRegistry::get().buildHooks()) {
    hook(qec->getIndex(), basename, spec);
  }
  for (const auto& hook : IndexExtensionRegistry::get().loadHooks()) {
    hook(impl, basename);
  }
  qec->setLocatedTriplesForEvaluation(
      impl.deltaTriplesManager().getCurrentLocatedTriplesSharedState());
  for (auto* suffix :
       {".vec.embbig.meta", ".vec.embbig.keys", ".vec.embbig.rowmap",
        ".vec.embbig.data", ".vec.embbig.iris", ".vec.embbig.hnsw",
        ".input.npy", ".input.iris"}) {
    std::error_code ec;
    std::filesystem::remove(basename + suffix, ec);
  }
  return qec;
}
}  // namespace

// _____________________________________________________________________________
// The flagship query of the design doc: enumerate the entities by their
// natural pattern, BIND the distance to an explicit query vector, drop
// non-members via FILTER(BOUND), and ORDER BY distance -- filtered top-k
// vector search from QLever's own operators, no SERVICE, no HTTP. (The LIMIT
// of the `... LIMIT 3` idiom is applied lazily during export, so the ranking
// is asserted on the ordered table.)
TEST(VectorIndexPayloadE2E, rankEntitiesByDistanceToQueryVector) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                         "BIND(vec:distance(vidx:emb, ?e, \"1,0,0\") AS ?d) "
                         "FILTER(BOUND(?d)) } ORDER BY ?d LIMIT 3");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  // `<nomember>` (no vector -> UNDEF) is dropped by FILTER(BOUND(?d)); the
  // four indexed entities remain, ranked by cosine distance to [1,0,0].
  ASSERT_EQ(table.numRows(), 4u);
  // The two exact matches <a> and <c> (distance ~0) rank first, in either
  // order (they are an exact tie).
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<a>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<c>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  // Then <b> (distance 1 - 0.6 = 0.4), then the orthogonal <d> (distance 1).
  EXPECT_EQ(table(2, eCol), getId("<b>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 0.4, 1e-3);
  EXPECT_EQ(table(3, eCol), getId("<d>"));
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
  // The ranking is monotone in the distance.
  EXPECT_LE(table(1, dCol).getDouble(), table(2, dCol).getDouble());
  EXPECT_LT(table(2, dCol).getDouble(), table(3, dCol).getDouble());
}

// _____________________________________________________________________________
// The index itself is a queryable RDF resource (design doc, section "idx:
// metadata triples"): at load time the hook auto-materializes thin metadata
// triples on `<.../vectorSearch/index/emb>` as delta triples, so the space is
// introspectable in plain SPARQL -- dimension, metric, precision, and member
// count, all matching the fixture's `.npy` bundle (4 vectors of dimension 3,
// cosine metric, stored as f32).
TEST(VectorIndexPayloadE2E, indexMetadataTriplesAreQueryable) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet = planQuery(
      qec,
      "SELECT ?dim ?metric ?precision ?count WHERE { "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/emb> "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/dimension> ?dim ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/metric> ?metric ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision> ?precision ; "
      "<https://qlever.cs.uni-freiburg.de/vectorSearch/count> ?count }");
  auto result = qet.getResult();
  const IdTable& table = result->idTable();

  // Exactly one metadata record for the index.
  ASSERT_EQ(table.numRows(), 1u);
  // The metadata literals live in the delta triples' local vocab.
  auto getLiteral = [&](std::string_view varName) {
    Id id = table(0, qet.getVariableColumn(Variable{std::string{varName}}));
    EXPECT_EQ(id.getDatatype(), Datatype::LocalVocabIndex)
        << varName << " should be a (local vocab) literal";
    return id.getLocalVocabIndex()->toStringRepresentation();
  };
  EXPECT_EQ(table(0, qet.getVariableColumn(Variable{"?dim"})),
            Id::makeFromInt(3));
  EXPECT_EQ(getLiteral("?metric"), "\"cosine\"");
  EXPECT_EQ(getLiteral("?precision"), "\"f32\"");
  EXPECT_EQ(table(0, qet.getVariableColumn(Variable{"?count"})),
            Id::makeFromInt(4));
}

// _____________________________________________________________________________
// An entity of the knowledge graph WITHOUT a vector in the index evaluates to
// UNDEF (the design's implicit membership filter), not an error and not a
// number; every indexed entity gets a real double.
TEST(VectorIndexPayloadE2E, nonMemberDistanceIsUndef) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                         "BIND(vec:distance(vidx:emb, ?e, \"1,0,0\") AS ?d) }");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  // All five entities (including the vectorless `<nomember>`).
  ASSERT_EQ(table.numRows(), 5u);
  bool sawNonMember = false;
  for (size_t r = 0; r < table.numRows(); ++r) {
    if (table(r, eCol) == getId("<nomember>")) {
      EXPECT_TRUE(table(r, dCol).isUndefined());
      sawNonMember = true;
    } else {
      EXPECT_EQ(table(r, dCol).getDatatype(), Datatype::Double)
          << "row " << r << " should have a real distance";
    }
  }
  EXPECT_TRUE(sawNonMember);
}

// _____________________________________________________________________________
// The sorted-input fast path end to end: when the per-row entity source is a
// bare variable bound by an index scan (so the entity column is the input's
// leading sort key), `vec:distance` takes the merge-walk fast path -- whose
// result must be BIT-IDENTICAL to the per-row primitive `(*computer)(entity)`.
// The same query with the entity wrapped in `COALESCE` (no longer a bare
// variable) declines the fast path and runs the per-row fallback; both must
// equal the reference and each other.
TEST(VectorIndexPayloadE2E, sortedFastPathMatchesPerRowReference) {
  auto* qec = qecWithPayloadIndex();
  auto vidx = qlever::vector::getVectorIndex(qec->getIndex(), "emb");
  ASSERT_TRUE(vidx);
  auto computer = vidx->makeDistanceComputer(std::vector<float>{1.f, 0.f, 0.f});

  // Run `query`, and assert every row's `?d` equals the reference distance-Id
  // computed for that row's `?e` via the per-row primitive. Returns the `?d`
  // column so callers can compare runs for determinism.
  auto runAndCheck = [&](const std::string& query) {
    QueryExecutionTree qet = planQuery(qec, std::string{PREFIX} + query);
    auto result = qet.getResult();
    size_t eCol = qet.getVariableColumn(Variable{"?e"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    EXPECT_EQ(table.numRows(), 5u);  // all five entities, incl. <nomember>.
    std::vector<std::pair<uint64_t, Id>> byEntity;
    for (size_t r = 0; r < table.numRows(); ++r) {
      Id e = table(r, eCol);
      Id expected = qlever::vector::distanceToValueId(computer(e));
      EXPECT_EQ(table(r, dCol), expected) << "row " << r;
      byEntity.emplace_back(e.getBits(), table(r, dCol));
    }
    std::sort(byEntity.begin(), byEntity.end());
    return byEntity;
  };

  // Fast path, constant on the RIGHT.
  auto fastRight = runAndCheck(
      "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
      "BIND(vec:distance(vidx:emb, ?e, \"1,0,0\") AS ?d) }");
  // Fast path, constant on the LEFT (metrics are symmetric; still fast path).
  auto fastLeft = runAndCheck(
      "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
      "BIND(vec:distance(vidx:emb, \"1,0,0\", ?e) AS ?d) }");
  // Fallback: COALESCE(?e) is not a bare variable, so the detection declines
  // and the per-row path runs.
  auto fallback = runAndCheck(
      "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
      "BIND(vec:distance(vidx:emb, \"1,0,0\", COALESCE(?e)) AS ?d) }");

  // Determinism: a second run of the fast-path query is identical.
  auto fastRightAgain = runAndCheck(
      "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
      "BIND(vec:distance(vidx:emb, ?e, \"1,0,0\") AS ?d) }");

  // Fast path (both orders), fallback, and the repeat all agree per entity.
  EXPECT_EQ(fastRight, fastLeft);
  EXPECT_EQ(fastRight, fallback);
  EXPECT_EQ(fastRight, fastRightAgain);
}

// _____________________________________________________________________________
// Entity<->entity distance with BOTH sources constant: `vec:distance` looks up
// both stored vectors ("how similar are <a> and <c>?"). The constant result
// broadcasts over the enumerated rows. <a> and <c> have identical stored
// vectors (distance ~0); <a> and <d> are orthogonal (cosine distance ~1); a
// pair involving the vectorless <nomember> is UNDEF.
TEST(VectorIndexPayloadE2E, entityToEntityDistanceByStoredVectors) {
  auto* qec = qecWithPayloadIndex();
  auto constantDistance = [&](std::string_view pair) {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                           "BIND(vec:distance(vidx:emb, " +
                           std::string{pair} + ") AS ?d) }");
    auto result = qet.getResult();
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    EXPECT_EQ(table.numRows(), 5u);
    return table(0, dCol);
  };
  Id identical = constantDistance("<a>, <c>");
  ASSERT_EQ(identical.getDatatype(), Datatype::Double);
  EXPECT_NEAR(identical.getDouble(), 0.0, 1e-4);
  Id orthogonal = constantDistance("<a>, <d>");
  ASSERT_EQ(orthogonal.getDatatype(), Datatype::Double);
  EXPECT_NEAR(orthogonal.getDouble(), 1.0, 1e-3);
  EXPECT_TRUE(constantDistance("<a>, <nomember>").isUndefined());
}

// _____________________________________________________________________________
// Entity<->entity with BOTH sources per-row variables: a variable pair join
// ranks all pairs correctly. Over the 4 indexed entities {a, b, c, d} with
// a = c = e0, b at cosine distance 0.4 from e0, and d orthogonal, the 16
// member pairs split into 6 identical pairs (distance ~0: (a,a), (a,c), (c,a),
// (c,c), (b,b), (d,d)), 4 pairs at ~0.4 (a/c x b, both directions), and 6
// orthogonal pairs at ~1. Pairs involving <nomember> are UNDEF and dropped by
// FILTER(BOUND).
TEST(VectorIndexPayloadE2E, variablePairDistanceRanksAllPairs) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?a ?b ?d WHERE { ?a <is-a> <Item> . ?b <is-a> <Item> . "
               "BIND(vec:distance(vidx:emb, ?a, ?b) AS ?d) "
               "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();

  // 5 x 5 pairs minus the 9 involving the vectorless <nomember>.
  ASSERT_EQ(table.numRows(), 16u);
  for (size_t r = 0; r < 6; ++r) {
    EXPECT_NEAR(table(r, dCol).getDouble(), 0.0, 1e-4) << "row " << r;
  }
  for (size_t r = 6; r < 10; ++r) {
    EXPECT_NEAR(table(r, dCol).getDouble(), 0.4, 1e-3) << "row " << r;
  }
  for (size_t r = 10; r < 16; ++r) {
    EXPECT_NEAR(table(r, dCol).getDouble(), 1.0, 1e-3) << "row " << r;
  }
}

// _____________________________________________________________________________
// A PER-ROW float-list string source (exactly the shape `vec:embed` produces
// per row): each row's query vector comes from a string-valued expression, is
// parsed per row, and rows whose entity has no vector stay UNDEF.
TEST(VectorIndexPayloadE2E, perRowFloatListStringSource) {
  auto* qec = qecWithPayloadIndex();
  // <b> is compared against its own vector (distance ~0); everything else
  // against <d>'s vector [0,0,1] (so <d> ~0, <a>/<c> ~1).
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
               "BIND(IF(STR(?e) = \"b\", \"0.6,0.8,0\", \"0,0,1\") AS ?q) "
               "BIND(vec:distance(vidx:emb, ?e, ?q) AS ?d) "
               "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  ASSERT_EQ(table.numRows(), 4u);  // <nomember> dropped via FILTER(BOUND).
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<b>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<d>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(2, dCol).getDouble(), 1.0, 1e-3);
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
}

// _____________________________________________________________________________
// `vec:embed` on an index WITHOUT a configured `embeddingUrl` fails with a
// clear error, also when composed into `vec:distance` (the text-search
// idiom). The LIVE embed path needs an embedding endpoint and is therefore
// not end-to-end testable here; `vec:embed`'s output SHAPE (a float-list
// string) is what the inline-string and per-row-string tests above exercise.
TEST(VectorIndexPayloadE2E, embedWithoutEndpointThrows) {
  auto* qec = qecWithPayloadIndex();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                         "BIND(vec:distance(vidx:emb, ?e, "
                         "vec:embed(vidx:emb, \"a red bicycle\")) AS ?d) }");
  AD_EXPECT_THROW_WITH_MESSAGE(
      qet.getResult(), ::testing::HasSubstr("has no embeddingUrl configured"));
}

// _____________________________________________________________________________
// The bf16 storage scalar with an fp32 INPUT: the `.npy` matrix is fp32, but
// `"scalar": "bf16"` stores every vector as 2-byte bf16 (the native `<V2`
// bf16 input is `nativeBf16NpyInputResolvesAndRanks` below). The fixture's
// components are exactly bf16-representable, so the f32 -> bf16 truncation is
// lossless and `vec:distance` ranks exactly like an fp32 store: identical
// vectors at distance ~0, orthogonal at cosine distance ~1.
TEST(VectorIndexPayloadE2E, bf16ScalarStoresTwoBytesAndRanksCorrectly) {
  auto [qec, dataFileSize] = qecWithBf16Index();

  // Proof that the flat store holds 2-byte bf16, not 4-byte fp32: the
  // `.vec.embbf16.data` file must fit the bf16 payload plus at most one page
  // of `MmapVector` rounding and its 32-byte trailer -- strictly less than
  // the raw fp32 payload alone would need.
  const std::uintmax_t bf16Payload = BF16_NUM_VECTORS * BF16_DIM * 2;
  EXPECT_GE(dataFileSize, bf16Payload);
  EXPECT_LE(dataFileSize,
            bf16Payload + static_cast<std::uintmax_t>(getpagesize()) + 32);
  EXPECT_LT(dataFileSize, BF16_NUM_VECTORS * BF16_DIM * 4);

  // The persisted metadata agrees: the auto-materialized `precision` triple
  // of the index says "bf16".
  {
    QueryExecutionTree qet = planQuery(
        qec,
        "SELECT ?precision WHERE { "
        "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/embbf16> "
        "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision> "
        "?precision }");
    auto result = qet.getResult();
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 1u);
    Id id = table(0, qet.getVariableColumn(Variable{"?precision"}));
    ASSERT_EQ(id.getDatatype(), Datatype::LocalVocabIndex);
    EXPECT_EQ(id.getLocalVocabIndex()->toStringRepresentation(), "\"bf16\"");
  }

  // Ranking against the query point e0 (all components bf16-exact, so the
  // only imprecision is the bf16 arithmetic of the distance kernel itself).
  std::string queryVector = "1";
  for (size_t i = 1; i < BF16_DIM; ++i) {
    queryVector += ",0";
  }
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <Bf16Item> . "
                         "BIND(vec:distance(vidx:embbf16, ?e, \"" +
                         queryVector +
                         "\") AS ?d) "
                         "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  ASSERT_EQ(table.numRows(), 4u);
  // The exact matches <a> and <c> (identical bf16 rows -> distance ~0) rank
  // first, in either order.
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<a>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<c>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  // Then <b> at 45 degrees (cosine distance 1 - 1/sqrt(2)), then the
  // orthogonal <d> at distance 1 (small tolerance for the bf16 kernel).
  EXPECT_EQ(table(2, eCol), getId("<b>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 1.0 - 1.0 / std::sqrt(2.0), 1e-2);
  EXPECT_EQ(table(3, eCol), getId("<d>"));
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
}

// _____________________________________________________________________________
// A NATIVE bf16 `.npy` input: the matrix dtype is `<V2` (ml_dtypes.bfloat16),
// 2 bytes per scalar, so the reader must use the 2-byte row stride and decode
// each little-endian bf16 bit pattern -- a wrong stride would mis-slice every
// row after the first. The sidecar uses bare IRIs, and all four rows resolve
// (FILTER(BOUND) keeps 4 rows). With `"scalar": "bf16"` and exactly
// bf16-representable components the ranking is exact: the identical vectors
// <a>/<c> at distance ~0, <b> at 45 degrees (1 - 1/sqrt(2)), <d> orthogonal
// (~1).
TEST(VectorIndexPayloadE2E, nativeBf16NpyInputResolvesAndRanks) {
  auto* qec = qecWithNativeBf16Index();
  QueryExecutionTree qet = planQuery(
      qec, std::string{PREFIX} +
               "SELECT ?e ?d WHERE { ?e <is-a> <NativeBf16Item> . "
               "BIND(vec:distance(vidx:embnative, ?e, \"1,0,0,0\") AS ?d) "
               "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t eCol = qet.getVariableColumn(Variable{"?e"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  // All four bare-IRI rows of the `<V2` bundle resolved to entities.
  ASSERT_EQ(table.numRows(), 4u);
  // The two rows identical to the query point rank first, in either order.
  std::set<uint64_t> nearest{table(0, eCol).getBits(),
                             table(1, eCol).getBits()};
  EXPECT_TRUE(nearest.contains(getId("<a>").getBits()));
  EXPECT_TRUE(nearest.contains(getId("<c>").getBits()));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.0, 1e-4);
  // Then <b> at 45 degrees, then the orthogonal <d>.
  EXPECT_EQ(table(2, eCol), getId("<b>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 1.0 - 1.0 / std::sqrt(2.0), 1e-2);
  EXPECT_EQ(table(3, eCol), getId("<d>"));
  EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
}

// _____________________________________________________________________________
// `vec:vector` fetches an entity's STORED vector as a TYPED query-vector
// literal `"floats"^^<.../vec/MODEL/PRECISION>` carrying the index's embedding
// space (here: model "clip", the f32 storage scalar) -- the exact form
// `vec:embed` also returns, and the one `vec:distance` validates.
TEST(VectorIndexPayloadE2E, vectorReturnsTypedLiteralOfIndexSpace) {
  auto* qec = qecWithCrossIndexes();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?v WHERE { ?a <is-a> <AItem> . "
                         "BIND(vec:vector(vidx:idxa, <a1>) AS ?v) }");
  auto result = qet.getResult();
  const IdTable& table = result->idTable();
  ASSERT_EQ(table.numRows(), 4u);
  Id id = table(0, qet.getVariableColumn(Variable{"?v"}));
  ASSERT_EQ(id.getDatatype(), Datatype::LocalVocabIndex);
  EXPECT_EQ(id.getLocalVocabIndex()->toStringRepresentation(),
            "\"1,0,0\"^^<" + std::string{VEC_DT_CLIP_F32} + ">");
}

// _____________________________________________________________________________
// The typed literal from `vec:vector` is accepted by `vec:distance` on the
// SAME index (identical space -> the validation passes) and ranks correctly:
// distance to <a1>'s own stored vector [1,0,0] gives <a1> ~0, <a2> ~0.4,
// <a3> ~1; the vectorless <a4> stays UNDEF and is filtered.
TEST(VectorIndexPayloadE2E, typedVectorLiteralAcceptedBySameIndex) {
  auto* qec = qecWithCrossIndexes();
  QueryExecutionTree qet =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?a ?d WHERE { ?a <is-a> <AItem> . "
                         "BIND(vec:distance(vidx:idxa, ?a, "
                         "vec:vector(vidx:idxa, <a1>)) AS ?d) "
                         "FILTER(BOUND(?d)) } ORDER BY ?d");
  auto result = qet.getResult();
  size_t aCol = qet.getVariableColumn(Variable{"?a"});
  size_t dCol = qet.getVariableColumn(Variable{"?d"});
  const IdTable& table = result->idTable();
  auto getId = makeGetId(qec->getIndex());

  ASSERT_EQ(table.numRows(), 3u);
  EXPECT_EQ(table(0, aCol), getId("<a1>"));
  EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
  EXPECT_EQ(table(1, aCol), getId("<a2>"));
  EXPECT_NEAR(table(1, dCol).getDouble(), 0.4, 1e-3);
  EXPECT_EQ(table(2, aCol), getId("<a3>"));
  EXPECT_NEAR(table(2, dCol).getDouble(), 1.0, 1e-3);
}

// _____________________________________________________________________________
// CROSS-INDEX distance between two indices that share an embedding space
// (same model "clip", same precision f32, same dimension 3): a vector fetched
// from `idxb` via `vec:vector` is validated and USED by `vec:distance` on
// `idxa`. Constant form first (<b1>'s vector [1,0,0] against all AItems),
// then the fully per-row pair form.
TEST(VectorIndexPayloadE2E, crossIndexDistanceSameSpaceRanks) {
  auto* qec = qecWithCrossIndexes();
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?d WHERE { ?a <is-a> <AItem> . "
                           "BIND(vec:distance(vidx:idxa, ?a, "
                           "vec:vector(vidx:idxb, <b1>)) AS ?d) "
                           "FILTER(BOUND(?d)) } ORDER BY ?d");
    auto result = qet.getResult();
    size_t aCol = qet.getVariableColumn(Variable{"?a"});
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    auto getId = makeGetId(qec->getIndex());
    ASSERT_EQ(table.numRows(), 3u);
    EXPECT_EQ(table(0, aCol), getId("<a1>"));
    EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
    EXPECT_EQ(table(1, aCol), getId("<a2>"));
    EXPECT_NEAR(table(1, dCol).getDouble(), 0.4, 1e-3);
    EXPECT_EQ(table(2, aCol), getId("<a3>"));
    EXPECT_NEAR(table(2, dCol).getDouble(), 1.0, 1e-3);
  }
  {
    // Per-row on BOTH sides: every (AItem, BItem) pair, ?b's vector fetched
    // from idxb per row and measured in idxa. Cosine distances: (a1,b1)=0,
    // (a2,b2)=0.2, (a2,b1)=0.4, and three orthogonal pairs at 1.
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?b ?d WHERE { "
                           "?a <is-a> <AItem> . ?b <is-a> <BItem> . "
                           "BIND(vec:distance(vidx:idxa, ?a, "
                           "vec:vector(vidx:idxb, ?b)) AS ?d) "
                           "FILTER(BOUND(?d)) } ORDER BY ?d");
    auto result = qet.getResult();
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    // 3 vectored AItems x 2 BItems (the vectorless <a4> pairs are UNDEF).
    ASSERT_EQ(table.numRows(), 6u);
    EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
    EXPECT_NEAR(table(1, dCol).getDouble(), 0.2, 1e-3);
    EXPECT_NEAR(table(2, dCol).getDouble(), 0.4, 1e-3);
    for (size_t r = 3; r < 6; ++r) {
      EXPECT_NEAR(table(r, dCol).getDouble(), 1.0, 1e-3) << "row " << r;
    }
  }
}

// _____________________________________________________________________________
// CROSS-INDEX with a DIFFERENT `embeddingModel` ("other" vs "clip"): the
// typed vector is NOT comparable, although dimension and precision match.
// A per-row source makes those rows UNDEF; a constant source throws a clear
// error (mirroring the dimension error), because it is certainly a query
// mistake.
TEST(VectorIndexPayloadE2E, crossIndexModelMismatchIsUndefOrThrows) {
  auto* qec = qecWithCrossIndexes();
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?d WHERE { "
                           "?a <is-a> <AItem> . ?o <is-a> <OItem> . "
                           "BIND(vec:distance(vidx:idxa, ?a, "
                           "vec:vector(vidx:idxother, ?o)) AS ?d) }");
    auto result = qet.getResult();
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 4u);
    for (size_t r = 0; r < table.numRows(); ++r) {
      EXPECT_TRUE(table(r, dCol).isUndefined()) << "row " << r;
    }
  }
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?d WHERE { ?a <is-a> <AItem> . "
                           "BIND(vec:distance(vidx:idxa, ?a, "
                           "vec:vector(vidx:idxother, <o1>)) AS ?d) }");
    AD_EXPECT_THROW_WITH_MESSAGE(
        qet.getResult(),
        ::testing::HasSubstr("does not match vector index \"idxa\""));
  }
}

// _____________________________________________________________________________
// A typed query-vector literal with the WRONG PRECISION (f16 vs the index's
// f32 store) or the WRONG DIMENSION: per-row -> UNDEF for exactly those rows;
// constant -> a clear throw.
TEST(VectorIndexPayloadE2E, typedLiteralWrongPrecisionOrDimension) {
  auto* qec = qecWithCrossIndexes();
  auto getId = makeGetId(qec->getIndex());
  // Per row: <a1> gets a correctly-typed query vector, every other row a
  // wrong-precision one -> only <a1> is bound.
  auto boundOnlyA1 = [&](const std::string& wrongTyped) {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?d WHERE { ?a <is-a> <AItem> . "
                           "BIND(IF(STR(?a) = \"a1\", \"1,0,0\"^^<" +
                           std::string{VEC_DT_CLIP_F32} + ">, " + wrongTyped +
                           ") AS ?q) "
                           "BIND(vec:distance(vidx:idxa, ?a, ?q) AS ?d) "
                           "FILTER(BOUND(?d)) }");
    auto result = qet.getResult();
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 1u);
    EXPECT_EQ(table(0, qet.getVariableColumn(Variable{"?a"})), getId("<a1>"));
    EXPECT_NEAR(table(0, qet.getVariableColumn(Variable{"?d"})).getDouble(),
                0.0, 1e-4);
  };
  // Wrong precision (f16-typed floats against the f32 store).
  boundOnlyA1("\"1,0,0\"^^<" + std::string{VEC_DT_CLIP_F16} + ">");
  // Wrong dimension (2 floats against the 3-dimensional index).
  boundOnlyA1("\"1,0\"^^<" + std::string{VEC_DT_CLIP_F32} + ">");

  // Constant sources throw loudly instead.
  auto expectThrow = [&](const std::string& constantSource,
                         const std::string& messagePart) {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?d WHERE { ?a <is-a> <AItem> . "
                           "BIND(vec:distance(vidx:idxa, ?a, " +
                           constantSource + ") AS ?d) }");
    AD_EXPECT_THROW_WITH_MESSAGE(qet.getResult(),
                                 ::testing::HasSubstr(messagePart));
  };
  expectThrow("\"1,0,0\"^^<" + std::string{VEC_DT_CLIP_F16} + ">",
              "does not match vector index \"idxa\"");
  expectThrow("\"1,0\"^^<" + std::string{VEC_DT_CLIP_F32} + ">",
              "has dimension 2, but vector index \"idxa\" has dimension 3");
}

// _____________________________________________________________________________
// `vec:vector` of an entity that is NOT a member of the index -> UNDEF, both
// for a constant entity (<b1> lives in idxb, not idxa) and per row (<a4> is
// an AItem without a vector).
TEST(VectorIndexPayloadE2E, vectorOfNonMemberIsUndef) {
  auto* qec = qecWithCrossIndexes();
  auto getId = makeGetId(qec->getIndex());
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?v WHERE { ?a <is-a> <AItem> . "
                           "BIND(vec:vector(vidx:idxa, <b1>) AS ?v) }");
    auto result = qet.getResult();
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 4u);
    EXPECT_TRUE(table(0, qet.getVariableColumn(Variable{"?v"})).isUndefined());
  }
  {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?a ?v WHERE { ?a <is-a> <AItem> . "
                           "BIND(vec:vector(vidx:idxa, ?a) AS ?v) }");
    auto result = qet.getResult();
    size_t aCol = qet.getVariableColumn(Variable{"?a"});
    size_t vCol = qet.getVariableColumn(Variable{"?v"});
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 4u);
    bool sawNonMember = false;
    for (size_t r = 0; r < table.numRows(); ++r) {
      if (table(r, aCol) == getId("<a4>")) {
        EXPECT_TRUE(table(r, vCol).isUndefined());
        sawNonMember = true;
      } else {
        EXPECT_EQ(table(r, vCol).getDatatype(), Datatype::LocalVocabIndex)
            << "row " << r << " should have a typed vector literal";
      }
    }
    EXPECT_TRUE(sawNonMember);
  }
}

// _____________________________________________________________________________
// The model check is SKIPPED when either side declares no model, so a
// vector-only index (the "emb" fixture has no `embeddingModel`) still accepts
// any typed query vector of matching precision and dimension -- and its own
// `vec:vector` output is typed with an EMPTY model segment (`.../vec//f32`)
// and round-trips through `vec:distance` on the same index.
TEST(VectorIndexPayloadE2E, modellessIndexSkipsModelCheck) {
  auto* qec = qecWithPayloadIndex();
  auto getId = makeGetId(qec->getIndex());
  {
    // A "clip"-typed constant query vector against the model-less index:
    // comparable (one side has no model), ranks like the flagship test.
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?e ?d WHERE { ?e <is-a> <Item> . "
                           "BIND(vec:distance(vidx:emb, ?e, \"1,0,0\"^^<" +
                           std::string{VEC_DT_CLIP_F32} +
                           ">) AS ?d) FILTER(BOUND(?d)) } ORDER BY ?d");
    auto result = qet.getResult();
    size_t dCol = qet.getVariableColumn(Variable{"?d"});
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 4u);
    EXPECT_NEAR(table(0, dCol).getDouble(), 0.0, 1e-4);
    EXPECT_NEAR(table(3, dCol).getDouble(), 1.0, 1e-3);
  }
  {
    // The model-less index's own `vec:vector` literal: empty model segment,
    // accepted right back by the same index.
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?v ?d WHERE { ?e <is-a> <Item> . "
                           "BIND(vec:vector(vidx:emb, <a>) AS ?v) "
                           "BIND(vec:distance(vidx:emb, <c>, ?v) AS ?d) }");
    auto result = qet.getResult();
    const IdTable& table = result->idTable();
    ASSERT_EQ(table.numRows(), 5u);
    Id v = table(0, qet.getVariableColumn(Variable{"?v"}));
    ASSERT_EQ(v.getDatatype(), Datatype::LocalVocabIndex);
    EXPECT_EQ(v.getLocalVocabIndex()->toStringRepresentation(),
              "\"1,0,0\"^^<https://qlever.cs.uni-freiburg.de/vectorSearch/"
              "vec//f32>");
    // <a> and <c> have identical stored vectors -> distance ~0.
    Id d = table(0, qet.getVariableColumn(Variable{"?d"}));
    ASSERT_EQ(d.getDatatype(), Datatype::Double);
    EXPECT_NEAR(d.getDouble(), 0.0, 1e-4);
  }
}

// _____________________________________________________________________________
// The whole-column scan `BIND(vec:distance(idx, ?e, <constant query>))` over an
// index with more entities than the parallel-scan threshold exercises the
// multi-threaded `perRowAgainstConstant` path. Its result MUST be BIT-IDENTICAL
// to a serial reference computed with the same `DistanceComputer` in a plain
// loop (proving the parallel path is race-free and deterministic, not merely
// numerically close), and two runs MUST agree exactly.
TEST(VectorIndexPayloadE2E, parallelPerRowScanMatchesSerialReference) {
  auto* qec = qecWithLargeIndex();

  // The exact same query the SPARQL expression will see, and hence the exact
  // same `DistanceComputer` -- the reference distances below therefore differ
  // from the query's only if the parallel scan is wrong.
  auto vidx = qlever::vector::getVectorIndex(qec->getIndex(), "embbig");
  ASSERT_TRUE(vidx);
  ASSERT_EQ(vidx->numLiveVectors(), BIG_NUM_VECTORS);
  ASSERT_EQ(vidx->dimensions(), BIG_DIM);
  auto computer = vidx->makeDistanceComputer(BIG_QUERY);
  auto referenceDistance = [&computer](Id entity) {
    float d = computer(entity);
    return std::isnan(d) ? Id::makeUndefined()
                         : Id::makeFromDouble(static_cast<double>(d));
  };

  auto runScan = [&](std::string queryVector) {
    QueryExecutionTree qet =
        planQuery(qec, std::string{PREFIX} +
                           "SELECT ?e ?d WHERE { ?e <is-a> <BigItem> . "
                           "BIND(vec:distance(vidx:embbig, ?e, \"" +
                           queryVector + "\") AS ?d) }");
    return qet.getResult();
  };

  auto result = runScan(std::string{BIG_QUERY_STR});
  const IdTable& table = result->idTable();
  // All entities have a vector -> one bound row each; > 2048 so the parallel
  // path was taken, and (< 10'000) it was a single BIND evaluation block.
  ASSERT_EQ(table.numRows(), BIG_NUM_VECTORS);

  // Every parallel distance is bit-identical to the serial reference.
  QueryExecutionTree probe =
      planQuery(qec, std::string{PREFIX} +
                         "SELECT ?e ?d WHERE { ?e <is-a> <BigItem> . "
                         "BIND(vec:distance(vidx:embbig, ?e, \"" +
                         std::string{BIG_QUERY_STR} + "\") AS ?d) }");
  size_t eCol = probe.getVariableColumn(Variable{"?e"});
  size_t dCol = probe.getVariableColumn(Variable{"?d"});
  size_t numBound = 0;
  for (size_t r = 0; r < table.numRows(); ++r) {
    Id entity = table(r, eCol);
    Id reference = referenceDistance(entity);
    ASSERT_EQ(table(r, dCol), reference)
        << "row " << r << " diverges from the serial reference";
    ASSERT_EQ(table(r, dCol).getDatatype(), Datatype::Double) << "row " << r;
    ++numBound;
  }
  EXPECT_EQ(numBound, BIG_NUM_VECTORS);

  // Determinism: a second run yields the same rows in the same order with the
  // same distances (same index-scan order, no ORDER BY).
  auto result2 = runScan(std::string{BIG_QUERY_STR});
  const IdTable& table2 = result2->idTable();
  ASSERT_EQ(table2.numRows(), table.numRows());
  for (size_t r = 0; r < table.numRows(); ++r) {
    EXPECT_EQ(table2(r, eCol), table(r, eCol)) << "row " << r;
    EXPECT_EQ(table2(r, dCol), table(r, dCol)) << "row " << r;
  }
}
