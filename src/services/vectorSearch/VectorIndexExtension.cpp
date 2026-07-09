// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Self-registration of the vector index as a QLever *index extension*: a BUILD
// hook (run by `qlever index` after the main index is built, with URI->Id
// resolution available) and a LOAD hook (run at server start, after the vocab
// is loaded). This is the ONLY coupling to the index/build lifecycle, and it
// touches no core code -- everything lives in this folder. See
// `index/IndexExtension.h`.

#include "services/vectorSearch/VectorIndexExtension.h"

#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#if USEARCH_USE_NUMKONG
// Pulls in `numkong/capabilities.h`: `nk_capabilities_available()`,
// `nk_capabilities_compiled()`, `nk_uses_dynamic_dispatch()`, and the
// `nk_cap_*_k` bits (the dispatch tables come from the linked
// `numkongDispatch` library).
#include <numkong/numkong.h>
#endif

#include "absl/strings/str_cat.h"
#include "backports/StartsWithAndEndsWith.h"
#include "backports/algorithm.h"
#include "global/Constants.h"
#include "global/IdTriple.h"
#include "index/DeltaTriples.h"
#include "index/Index.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "index/LocalVocab.h"
#include "index/StringSortComparator.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "rdfTypes/Literal.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorIndexFormat.h"
#include "services/vectorSearch/VectorInputReader.h"
#include "util/Exception.h"
#include "util/Log.h"
#include "util/json.h"
#include "util/jthread.h"

namespace qlever::vector {

// ____________________________________________________________________________
std::shared_ptr<const VectorIndex> getVectorIndex(const Index& index,
                                                  const std::string& name) {
  std::shared_ptr<void> ext =
      index.getImpl().getExtension(std::string{VECTOR_EXTENSION_NAME});
  if (!ext) {
    return nullptr;
  }
  auto collection =
      std::static_pointer_cast<const VectorIndexCollection>(std::move(ext));
  const VectorIndex* idx = collection->get(name);
  if (idx == nullptr) {
    return nullptr;
  }
  // Aliasing constructor: shares ownership of the collection, points at the
  // contained index.
  return std::shared_ptr<const VectorIndex>{std::move(collection), idx};
}

// ____________________________________________________________________________
std::string numkongActiveIsaString([[maybe_unused]] uint64_t capabilities) {
#if USEARCH_USE_NUMKONG
  std::string out;
  auto append = [&](uint64_t bit, std::string_view name) {
    if ((capabilities & bit) != 0) {
      if (!out.empty()) {
        out.push_back(' ');
      }
      out.append(name);
    }
  };
  // x86.
  append(nk_cap_haswell_k, "haswell/avx2");
  append(nk_cap_skylake_k, "skylake/avx512");
  append(nk_cap_icelake_k, "icelake/avx512-vnni");
  append(nk_cap_genoa_k, "genoa/avx512-bf16");
  append(nk_cap_sapphire_k, "sapphire/avx512-fp16");
  append(nk_cap_sapphireamx_k, "amx");
  append(nk_cap_graniteamx_k, "amx-fp16");
  append(nk_cap_turin_k, "turin");
  // ARM.
  append(nk_cap_neon_k, "neon");
  append(nk_cap_neonhalf_k, "neon-fp16");
  append(nk_cap_neonbfdot_k, "neon-bfdot");
  append(nk_cap_sve_k, "sve");
  append(nk_cap_sve2_k, "sve2");
  if (out.empty()) {
    out = "serial(scalar)";
  }
  return out;
#else
  // Built without NumKong (QLEVER_VECTOR_SEARCH_SIMD=OFF): there is never a
  // SIMD kernel, whatever the bitmask claims.
  return "serial(scalar)";
#endif
}

namespace {

// Removes a set of files on destruction unless `dismiss()` was called; used so
// that a failed remap does not leave `.tmp` outputs behind.
class RemapTmpCleanup {
 public:
  explicit RemapTmpCleanup(std::vector<std::string> paths)
      : paths_{std::move(paths)} {}
  void dismiss() { paths_.clear(); }
  ~RemapTmpCleanup() {
    for (const auto& path : paths_) {
      std::error_code ec;
      std::filesystem::remove(path, ec);
    }
  }

 private:
  std::vector<std::string> paths_;
};

// How many input rows are read and resolved per batch.
constexpr size_t RESOLVE_BATCH_SIZE = 16384;

// The size of the loaded knowledge-graph vocabulary; used as the fingerprint
// that binds a vector index to the main-index build it was created from.
uint64_t vocabFingerprint(const IndexImpl& impl) {
  return static_cast<uint64_t>(impl.getVocab().size());
}

// The knowledge-graph vocabulary's collation fingerprint. Entity ids are
// `VocabIndex` positions in the collation-sorted vocabulary, so this is
// persisted at build/remap time and re-checked at load: a mismatch means the
// collation changed, which reassigns every id and invalidates the persisted
// id->row mapping. The load hook therefore SKIPS such an index (forcing a
// remap), like the vocab-size guard -- it is a correctness gate, not a
// performance hint.
std::string collationFingerprint(const IndexImpl& impl) {
  return impl.getVocab().getLocaleManager().getCollationIdentifier();
}

size_t effectiveThreads(uint32_t configured) {
  return configured > 0 ? configured
                        : std::max(1u, std::thread::hardware_concurrency());
}

// Parse one `--service-index` JSON object entry into a build spec. Rejects
// unknown keys so that a typo does not silently fall back to a default.
VectorIndexBuildSpec parseSpec(const nlohmann::json& obj) {
  VectorIndexBuildSpec spec;
  try {
    spec.config_.name_ = obj.at("name").get<std::string>();
    spec.remap_ = obj.value("remap", false);
  } catch (const nlohmann::json::exception& e) {
    AD_THROW(std::string{"Malformed vectorSearch build spec: "} + e.what());
  }
  // The name is spliced into the on-disk filenames -- restrict it so it cannot
  // escape the index directory or collide with the file-pattern parsing.
  const std::string& name = spec.config_.name_;
  bool nameOk = !name.empty() && ql::ranges::all_of(name, [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_';
  });
  if (!nameOk) {
    AD_THROW("Invalid vector index name \"" + name +
             "\"; use only letters, digits, '-', and '_'.");
  }

  if (spec.remap_) {
    // A remap reuses the existing on-disk configuration; only the entity
    // mapping is recomputed.
    static constexpr std::array remapKeys{"name", "remap", "buildThreads"};
    for (const auto& [key, value] : obj.items()) {
      if (ql::ranges::find(remapKeys, key) == remapKeys.end()) {
        AD_THROW("Unknown key \"" + key +
                 "\" in a vectorSearch remap spec. Known keys: name, remap, "
                 "buildThreads.");
      }
    }
    spec.config_.buildThreads_ = obj.value("buildThreads", uint32_t{0});
    return spec;
  }

  static constexpr std::array knownKeys{"name",
                                        "iris",
                                        "npy",
                                        "dimensions",
                                        "metric",
                                        "scalar",
                                        "rerank",
                                        "csls",
                                        "cslsNeighbors",
                                        "cslsR",
                                        "hnsw",
                                        "hnswConnectivity",
                                        "hnswExpansionAdd",
                                        "hnswExpansionSearch",
                                        "buildThreads",
                                        "remap"};
  for (const auto& [key, value] : obj.items()) {
    if (ql::ranges::find(knownKeys, key) == knownKeys.end()) {
      AD_THROW("Unknown key \"" + key +
               "\" in a vectorSearch build spec. Known keys: name, iris, npy, "
               "dimensions, metric, scalar, rerank, csls, cslsNeighbors, "
               "cslsR, hnsw, hnswConnectivity, hnswExpansionAdd, "
               "hnswExpansionSearch, buildThreads, remap. "
               "(The query-time embedding endpoint/model and the RAM "
               "residency are serving concerns, set at server start via the "
               "QLEVER_VECTOR_SEARCH_ENDPOINTS environment variable, not at "
               "build time.)");
    }
  }
  // The only vector input is a `.npy` matrix plus its row-aligned `iris`
  // sidecar; both are required.
  if (!obj.contains("npy") || !obj.contains("iris")) {
    AD_THROW(
        "Each vectorSearch build spec needs both an `npy` matrix and a "
        "row-aligned `iris` list.");
  }
  try {
    spec.irisPath_ = obj.at("iris").get<std::string>();
    spec.npyPath_ = obj.at("npy").get<std::string>();
    spec.config_.dimensions_ = obj.value("dimensions", uint32_t{0});
    spec.config_.metric_ =
        vectorMetricFromString(obj.value("metric", std::string{"cosine"}));
    spec.config_.scalar_ =
        vectorScalarFromString(obj.value("scalar", std::string{"f32"}));
    // Two-layer quantize+rerank: an optional second storage precision for the
    // fine `.rerank.data` matrix. Empty (the default) = single-layer.
    if (std::string rerank = obj.value("rerank", std::string{});
        !rerank.empty()) {
      spec.config_.rerankScalar_ = vectorScalarFromString(rerank);
    }
    spec.config_.csls_ = obj.value("csls", false);
    spec.config_.cslsNeighbors_ = obj.value("cslsNeighbors", uint32_t{10});
    spec.config_.cslsRPath_ = obj.value("cslsR", std::string{});
    spec.config_.buildHnsw_ = obj.value("hnsw", true);
    // The `hnsw*` keys configure the MAIN query-time ANN graph only. The csls
    // r(d) self-kNN never touches that graph: the builder computes r(d) on
    // the FINE layer, by exact brute force or via its own recall-tuned
    // fine-layer graph (see `VectorIndexBuilder`), independent of these keys
    // and of `hnsw` itself.
    spec.config_.hnswConnectivity_ =
        obj.value("hnswConnectivity", uint32_t{16});
    spec.config_.hnswExpansionAdd_ =
        obj.value("hnswExpansionAdd", uint32_t{128});
    spec.config_.hnswExpansionSearch_ =
        obj.value("hnswExpansionSearch", uint32_t{64});
    spec.config_.buildThreads_ = obj.value("buildThreads", uint32_t{0});
  } catch (const nlohmann::json::exception& e) {
    AD_THROW(std::string{"Malformed vectorSearch build spec: "} + e.what());
  }
  // The `i8` store rescales every vector to a common magnitude (usearch's
  // dot-product-oriented int8 cast), which destroys the magnitude information
  // that `l2sq`/`innerProduct` depend on -- only `cosine` is meaningful.
  if (spec.config_.scalar_ == VectorScalar::I8 &&
      spec.config_.metric_ != VectorMetric::Cosine) {
    AD_THROW(
        "The `i8` scalar type normalizes each vector, so it only makes sense "
        "with `metric: cosine` (got `l2sq`/`innerProduct`). Use `f16` or "
        "`f32` for those metrics.");
  }
  // The `binary` store keeps only the SIGN bit of each component and ranks by
  // Hamming distance -- an angular proxy that carries no magnitude at all, so
  // like i8 it is cosine-only.
  if (spec.config_.scalar_ == VectorScalar::Binary &&
      spec.config_.metric_ != VectorMetric::Cosine) {
    AD_THROW(
        "The `binary` scalar type keeps only the sign bit of each component "
        "(ranked by Hamming distance, an angular proxy), so it only makes "
        "sense with `metric: cosine` (got `l2sq`/`innerProduct`). Use `f16` "
        "or `f32` for those metrics.");
  }
  // The rerank layer is the HIGH-precision layer that exact distances read;
  // a quantized i8/binary rerank matrix would defeat its purpose (and both
  // carry the cosine-only restriction on top).
  if (spec.config_.rerankScalar_ == VectorScalar::I8 ||
      spec.config_.rerankScalar_ == VectorScalar::Binary) {
    AD_THROW(
        "The `rerank` precision must be one of `bf16`, `f16`, or `f32` (the "
        "high-precision layer the rerank pass and `vec:distance` read); `" +
        toString(spec.config_.rerankScalar_.value()) +
        "` belongs in the `scalar` scan layer.");
  }
  // CSLS is cosine-specific: the query-time cut converts distances back to
  // similarities as `cos_sim = 1 - distance`, which only holds for the
  // cosine metric.
  if (spec.config_.csls_ && spec.config_.metric_ != VectorMetric::Cosine) {
    AD_THROW(
        "`csls` works in cosine-similarity space (the cut converts distances "
        "back via `cos_sim = 1 - distance`), so it requires `metric: cosine` "
        "(got `" +
        toString(spec.config_.metric_) + "`).");
  }
  // A `binary` store without a rerank layer serves only Hamming distances --
  // no exact cosine anywhere to compute the CSLS terms from.
  if (spec.config_.csls_ && spec.config_.scalar_ == VectorScalar::Binary &&
      !spec.config_.rerankScalar_.has_value()) {
    AD_THROW(
        "`csls` needs exact cosine distances, but `scalar: binary` without a "
        "`rerank` layer only serves Hamming distances. Add e.g. `\"rerank\": "
        "\"bf16\"`.");
  }
  // The csls sub-keys without `csls: true` would silently do nothing.
  if (!spec.config_.csls_ &&
      (obj.contains("cslsNeighbors") || obj.contains("cslsR"))) {
    AD_THROW(
        "The `cslsNeighbors`/`cslsR` keys require `\"csls\": true` in the "
        "same vectorSearch build spec.");
  }
  if (spec.config_.csls_ && spec.config_.cslsNeighbors_ == 0) {
    AD_THROW("`cslsNeighbors` must be a positive integer.");
  }
  // A binary index WITHOUT a rerank layer is allowed but every distance it
  // can serve is the Hamming sign-bit proxy -- point that out at build time.
  if (spec.config_.scalar_ == VectorScalar::Binary &&
      !spec.config_.rerankScalar_.has_value()) {
    AD_LOG_WARN
        << "Vector index \"" << spec.config_.name_
        << "\": `scalar: binary` without a `rerank` layer stores only the "
           "sign bits; ALL distances (searches, `vec:distance`, scores) are "
           "integer Hamming distances -- an approximate angular proxy, never "
           "exact cosine. Add e.g. `\"rerank\": \"bf16\"` for exact scores."
        << std::endl;
  }
  if (spec.config_.buildHnsw_ && spec.config_.hnswConnectivity_ < 2) {
    AD_THROW("`hnswConnectivity` must be at least 2.");
  }
  return spec;
}

// Resolve `iris[i]` (line numbers starting at `firstLineNumber`) against the
// knowledge graph's vocabulary, in parallel. Vocabulary lookups are read-only
// and independent, so this scales with the core count -- crucial for inputs
// with hundreds of millions of rows. Throws (with the line number) on lines
// that are not of the form `<...>`.
std::vector<std::optional<Id>> resolveBatch(
    const Index& index, const std::vector<std::string>& iris,
    uint64_t firstLineNumber, size_t numThreads) {
  std::vector<std::optional<Id>> out(iris.size());
  numThreads = std::max<size_t>(1, std::min(numThreads, iris.size()));
  std::atomic_flag failed = ATOMIC_FLAG_INIT;
  std::mutex errorMutex;
  std::string firstError;
  auto job = [&](size_t first, size_t last) {
    try {
      for (size_t i = first; i < last; ++i) {
        const std::string& iri = iris[i];
        if (iri.size() < 2 || !ql::starts_with(iri, "<") ||
            !ql::ends_with(iri, ">")) {
          throw std::runtime_error{
              "Line " + std::to_string(firstLineNumber + i) +
              " of the IRI input is not an IRI of the form <...>: \"" + iri +
              "\""};
        }
        TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(iri)};
        out[i] = tc.toValueId(index.getImpl());
      }
    } catch (const std::exception& e) {
      if (!failed.test_and_set()) {
        std::lock_guard lock{errorMutex};
        firstError = e.what()[0] == '\0' ? "unknown resolve error" : e.what();
      }
    } catch (...) {
      if (!failed.test_and_set()) {
        std::lock_guard lock{errorMutex};
        firstError = "non-standard exception while resolving IRIs";
      }
    }
  };
  if (numThreads <= 1) {
    job(0, iris.size());
  } else {
    std::vector<ad_utility::JThread> threads;
    size_t chunk = (iris.size() + numThreads - 1) / numThreads;
    for (size_t t = 0; t < numThreads; ++t) {
      size_t first = t * chunk;
      size_t last = std::min(iris.size(), first + chunk);
      if (first >= last) break;
      threads.emplace_back(job, first, last);
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }
  {
    std::lock_guard lock{errorMutex};
    if (!firstError.empty()) {
      AD_THROW(firstError);
    }
  }
  return out;
}

// Stream `(iri, vector)` rows from `reader` (the `.npy` input) into a builder,
// resolving the IRIs in parallel batches. Readers may yield IRIs bare or as
// `<...>` irirefs; bare ones are bracketed here, so the builder always stores
// (and `resolveBatch` always sees) the `<...>` form.
VectorIndexMetadata buildFromReader(const Index& index,
                                    const std::string& basename,
                                    VectorIndexConfig config,
                                    VectorInputReader& reader,
                                    uint64_t& resolved, uint64_t& skipped) {
  const size_t numThreads = effectiveThreads(config.buildThreads_);
  VectorIndexBuilder builder{basename, config};
  builder.setVocabSize(vocabFingerprint(index.getImpl()));
  builder.setCollationLocale(collationFingerprint(index.getImpl()));
  // The optional PRECOMPUTED csls r(d) column (the `cslsR` GPU path): one f32
  // per INPUT row, in the input's row order. Each value follows its row
  // through the resolve/skip below (and the builder's dedup/sort), so the
  // final `.csls` sidecar stays row-aligned with the store.
  std::vector<float> cslsRColumn;
  if (config.csls_ && !config.cslsRPath_.empty()) {
    cslsRColumn = readNpyFloatColumn(config.cslsRPath_);
    if (cslsRColumn.size() != reader.numRows()) {
      AD_THROW("The cslsR file " + config.cslsRPath_ + " has " +
               std::to_string(cslsRColumn.size()) +
               " values, but the vector input has " +
               std::to_string(reader.numRows()) +
               " rows; the two must be row-aligned.");
    }
    for (size_t i = 0; i < cslsRColumn.size(); ++i) {
      if (!std::isfinite(cslsRColumn[i])) {
        AD_THROW("Row " + std::to_string(i + 1) + " of the cslsR file " +
                 config.cslsRPath_ + " is not a finite number.");
      }
    }
  }
  const size_t dim = config.dimensions_;
  std::vector<std::string> iris;
  std::vector<float> vectors;  // batch, row-major
  uint64_t lineNumber = 1;
  bool done = false;
  while (!done) {
    iris.clear();
    vectors.clear();
    std::string iri;
    std::vector<float> vec;
    while (iris.size() < RESOLVE_BATCH_SIZE) {
      if (!reader.next(iri, vec)) {
        done = true;
        break;
      }
      // Guard against a reader that yields a per-row length disagreeing with
      // the configured dimension -- otherwise the row would be silently
      // mis-sliced into the wrong entity's vector.
      if (vec.size() != dim) {
        AD_THROW("Row " + std::to_string(lineNumber + iris.size()) +
                 " of the input has dimension " + std::to_string(vec.size()) +
                 ", but the index is configured with dimension " +
                 std::to_string(dim) + ".");
      }
      if (!ql::starts_with(iri, "<")) {
        iri = "<" + iri + ">";
      }
      iris.push_back(iri);
      vectors.insert(vectors.end(), vec.begin(), vec.end());
    }
    auto ids = resolveBatch(index, iris, lineNumber, numThreads);
    for (size_t i = 0; i < iris.size(); ++i) {
      if (ids[i].has_value()) {
        // The absolute (0-based) input row of this batch entry, indexing the
        // row-aligned cslsR column.
        const size_t inputRow = static_cast<size_t>(lineNumber - 1) + i;
        std::optional<float> cslsR =
            cslsRColumn.empty() ? std::nullopt
                                : std::optional<float>{cslsRColumn[inputRow]};
        builder.add(ids[i].value(), iris[i],
                    ql::span<const float>{vectors.data() + i * dim, dim},
                    cslsR);
        ++resolved;
      } else {
        ++skipped;
      }
    }
    lineNumber += iris.size();
  }
  return builder.build();
}

VectorIndexMetadata buildFromNpy(const Index& index,
                                 const std::string& basename,
                                 const VectorIndexBuildSpec& spec,
                                 uint64_t& resolved, uint64_t& skipped) {
  NpyVectorInputReader reader{spec.npyPath_, spec.irisPath_};
  VectorIndexConfig config = spec.config_;
  if (config.dimensions_ == 0) {
    config.dimensions_ = reader.dimensions();
  }
  if (config.dimensions_ != reader.dimensions()) {
    AD_THROW("The configured dimension (" + std::to_string(config.dimensions_) +
             ") does not match the .npy file (" +
             std::to_string(reader.dimensions()) + ").");
  }
  return buildFromReader(index, basename, config, reader, resolved, skipped);
}

// BUILD hook: build (or remap) every vector index requested under the
// "vectorSearch" key.
void buildHook(const Index& index, const std::string& basename,
               const nlohmann::json& fullConfig) {
  if (!fullConfig.contains(std::string{VECTOR_EXTENSION_NAME})) {
    return;
  }
  const auto& specs = fullConfig.at(std::string{VECTOR_EXTENSION_NAME});
  if (!specs.is_array()) {
    AD_THROW("The `vectorSearch` build config must be a JSON array.");
  }
  for (const auto& obj : specs) {
    VectorIndexBuildSpec spec = parseSpec(obj);
    if (spec.remap_) {
      auto [live, tombstones] = remapVectorIndex(
          index, basename, spec.config_.name_, spec.config_.buildThreads_);
      AD_LOG_INFO << "Vector index '" << spec.config_.name_ << "': remapped "
                  << live << " entities to the re-indexed knowledge graph, "
                  << tombstones << " tombstoned (no longer in the graph)."
                  << std::endl;
      continue;
    }
    uint64_t resolved = 0;
    uint64_t skipped = 0;
    VectorIndexMetadata meta =
        buildFromNpy(index, basename, spec, resolved, skipped);
    AD_LOG_INFO << "Vector index '" << spec.config_.name_ << "': indexed "
                << resolved << " vectors, skipped " << skipped
                << " (IRI not in the knowledge graph), HNSW="
                << (meta.hasHnsw_ ? "yes" : "no") << std::endl;
  }
}

// The fields of one loaded vector index that become its auto-materialized
// metadata triples (collected while the load hook iterates the indices).
struct IndexMetadataFields {
  std::string name_;
  uint64_t dimensions_;
  std::string metric_;
  std::string precision_;
  uint64_t count_;
  std::string model_;  // empty when the index has no embedding endpoint
};

// Auto-materialize the thin per-index metadata triples (see
// `docs/vector-index/index-payload-design.md`, section "idx: metadata
// triples") as DELTA triples, so that every loaded vector index is
// introspectable in plain SPARQL:
//   SELECT * { <.../vectorSearch/index/emb> ?p ?o }
// Runs once per server start and inserts a handful of triples per index (so
// cost is irrelevant). `writeToDiskAfterRequest` is false: the triples are
// re-materialized on every load anyway (re-insertion of an already inserted
// triple is a no-op), so persisted-update files need not carry them.
void materializeMetadataTriples(IndexImpl& impl,
                                const std::vector<IndexMetadataFields>& rows) {
  if (rows.empty()) {
    return;
  }
  // Locating delta triples requires the block metadata of all (loaded)
  // permutations; without them (`--only-pso-and-pos-permutations` or
  // `doNotLoadPermutations`) updates are unsupported anyway, so skip.
  if (impl.doNotLoadPermutations() || !impl.loadAllPermutations()) {
    AD_LOG_WARN << "Not materializing the metadata triples of the vector "
                   "indices: they require all permutations to be loaded."
                << std::endl;
    return;
  }
  try {
    impl.deltaTriplesManager().modify<void>(
        [&impl, &rows](DeltaTriples& deltaTriples) {
          // New IRIs and literals get their ids from this local vocab.
          // `insertTriples` re-homes the entries into the delta triples' own
          // local vocab (`rewriteLocalVocabEntriesAndBlankNodes`), so this one
          // only has to outlive the call.
          LocalVocab localVocab;
          auto toId = [&impl, &localVocab](TripleComponent tc) {
            return std::move(tc).toValueId(impl, localVocab);
          };
          using Iri = ad_utility::triple_component::Iri;
          using Literal = ad_utility::triple_component::Literal;
          const Id graph = toId(Iri::fromIriref(DEFAULT_GRAPH_IRI));
          DeltaTriples::Triples triples;
          for (const IndexMetadataFields& row : rows) {
            const Id subject = toId(Iri::fromIriref(
                absl::StrCat(VECTOR_METADATA_SUBJECT_PREFIX, row.name_, ">")));
            auto add = [&](std::string_view predicate, TripleComponent object) {
              triples.push_back(
                  IdTriple<0>{{subject, toId(Iri::fromIriref(predicate)),
                               toId(std::move(object)), graph}});
            };
            add(VECTOR_METADATA_DIMENSION_IRI,
                static_cast<int64_t>(row.dimensions_));
            add(VECTOR_METADATA_METRIC_IRI,
                Literal::literalWithoutQuotes(row.metric_));
            add(VECTOR_METADATA_PRECISION_IRI,
                Literal::literalWithoutQuotes(row.precision_));
            add(VECTOR_METADATA_COUNT_IRI, static_cast<int64_t>(row.count_));
            if (!row.model_.empty()) {
              add(VECTOR_METADATA_MODEL_IRI,
                  Literal::literalWithoutQuotes(row.model_));
            }
          }
          // `insertTriples` requires its input to be sorted.
          ql::ranges::sort(triples);
          auto handle = std::make_shared<
              DeltaTriples::CancellationHandle::element_type>();
          deltaTriples.insertTriples(std::move(handle), std::move(triples));
        },
        /*writeToDiskAfterRequest=*/false);
  } catch (const std::exception& e) {
    // Metadata triples are a convenience; never let them prevent the server
    // from starting.
    AD_LOG_WARN << "Could not materialize the metadata triples of the vector "
                   "indices: "
                << e.what() << std::endl;
  }
}

// Log ONCE per process (at the first vector-index load) whether NumKong's
// SIMD kernels are actually engaged -- i.e. which detected-AND-compiled ISAs
// the runtime dispatch can pick from -- or warn loudly when everything would
// run on the scalar fallback.
void logNumkongSimdOnce() {
  static std::once_flag flag;
  std::call_once(flag, [] {
#if USEARCH_USE_NUMKONG
    const nk_capability_t available = nk_capabilities_available();
    if ((available & ~static_cast<nk_capability_t>(nk_cap_serial_k)) == 0) {
      AD_LOG_WARN << "Vector search: NO SIMD ISA active -- NumKong is on the "
                     "scalar fallback; distance scans will be slow."
                  << std::endl;
      return;
    }
    AD_LOG_INFO << "Vector search SIMD: dynamic-dispatch="
                << (nk_uses_dynamic_dispatch() != 0 ? "on" : "off")
                << ", active ISA: " << numkongActiveIsaString(available)
                << std::endl;
#else
    AD_LOG_WARN << "Vector search: NO SIMD ISA active -- this binary was "
                   "built without NumKong (QLEVER_VECTOR_SEARCH_SIMD=OFF); "
                   "distance scans will be slow."
                << std::endl;
#endif
  });
}

// LOAD hook: memory-map every `<base>.vec.<name>.*` index and attach the
// collection to the `IndexImpl`. A broken or stale index is skipped with a
// warning instead of preventing the whole server from starting -- queries
// against it then fail with a clear "no such vector index" error.
void loadHook(IndexImpl& impl, const std::string& basename) {
  namespace fs = std::filesystem;
  fs::path basePath{basename};
  const std::string prefix = basePath.filename().string() + ".vec.";
  const std::string suffix = ".meta";
  fs::path dir =
      basePath.has_parent_path() ? basePath.parent_path() : fs::path{"."};
  std::error_code ec;
  if (!fs::exists(dir, ec)) {
    return;
  }
  auto collection = std::make_shared<VectorIndexCollection>();
  std::vector<IndexMetadataFields> metadataFields;
  // Runtime configuration of the embedding endpoints and RAM residency (see
  // `VECTOR_SEARCH_ENDPOINTS_ENV_VAR` in `VectorIndexExtension.h`): applied IN
  // MEMORY to every matching index below -- the "preload" field at `open`
  // time, the endpoint fields to the opened index. These are serving concerns
  // that are never persisted, so they are reapplied fresh at every server
  // start. Applied entries are erased, so what remains afterwards is a
  // typo/stale name to warn about.
  EmbeddingEndpointOverrides endpointOverrides =
      embeddingEndpointOverridesFromEnv();
  bool any = false;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const std::string fn = entry.path().filename().string();
    if (fn.size() <= prefix.size() + suffix.size() ||
        !ql::starts_with(fn, prefix) || !ql::ends_with(fn, suffix)) {
      continue;
    }
    const std::string name =
        fn.substr(prefix.size(), fn.size() - prefix.size() - suffix.size());
    // A "preload"/"preloadRerank" setting must be decided AT open time --
    // residency is applied when the index is opened -- unlike the
    // embedding-endpoint fields below, which mutate the already-opened index.
    // No setting (and an explicit "none") leaves the respective store
    // mmap-only, the default. The two layers are independent: "preload"
    // governs the coarse scan matrix, "preloadRerank" the fine rerank matrix
    // of a two-layer index (a no-op on a single-layer one). The entry is NOT
    // erased here: the endpoint fields still apply below, and an entry for an
    // index that fails to open must survive for the leftover warning at the
    // end.
    VectorIndex::Residency residency = VectorIndex::Residency::None;
    VectorIndex::Residency rerankResidency = VectorIndex::Residency::None;
    if (auto it = endpointOverrides.find(name); it != endpointOverrides.end()) {
      auto residencyFor = [&](const std::optional<std::string>& preload,
                              const char* what) {
        if (!preload.has_value()) {
          return VectorIndex::Residency::None;
        }
        auto r = VectorIndex::residencyFromString(preload.value());
        if (r != VectorIndex::Residency::None) {
          AD_LOG_INFO << "Vector index '" << name << "': " << what
                      << " RAM residency set to '" << preload.value()
                      << "' at startup (" << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                      << ")" << std::endl;
        }
        return r;
      };
      residency = residencyFor(it->second.preload_, "scan");
      rerankResidency = residencyFor(it->second.preloadRerank_, "rerank");
    }
    VectorIndex idx;
    try {
      idx.open(basename, name, residency, rerankResidency);
    } catch (const std::exception& e) {
      AD_LOG_WARN << "Skipping vector index '" << name << "': " << e.what()
                  << std::endl;
      continue;
    }
    if (idx.metadata().vocabSize_ != vocabFingerprint(impl)) {
      AD_LOG_WARN
          << "Skipping vector index '" << name
          << "': it was built against a different version of the knowledge "
             "graph (vocabulary size "
          << idx.metadata().vocabSize_ << " at build time vs. "
          << vocabFingerprint(impl)
          << " now), so its entity ids are no longer valid. Remap it with "
             "`--service-index '{\"vectorSearch\":[{\"name\":\""
          << name << "\",\"remap\":true}]}'` (cheap; reuses the vectors and "
          << "the HNSW graph)." << std::endl;
      continue;
    }
    // Collation guard: entity ids are `VocabIndex` positions in the
    // collation-sorted vocabulary, so a changed collation re-sorts the vocab
    // and reassigns every id -- WITHOUT changing the vocabulary size, so the
    // size guard above does not catch it. The persisted `.keys`/`.rowmap` then
    // map the new query-time ids to the wrong rows/entities. This is a
    // correctness problem (not merely a slower gather), so skip the index and
    // force a remap, exactly like the size mismatch above (a remap re-resolves
    // the immutable `.iris` against the current vocabulary).
    if (const std::string& built = idx.metadata().collationLocale_;
        !built.empty() && built != collationFingerprint(impl)) {
      AD_LOG_WARN << "Skipping vector index '" << name
                  << "': it was built with "
                  << "collation \"" << built
                  << "\", but the knowledge graph now uses \""
                  << collationFingerprint(impl)
                  << "\", so its entity ids are no longer valid. Remap it with "
                     "`--service-index '{\"vectorSearch\":[{\"name\":\""
                  << name
                  << "\",\"remap\":true}]}'` (cheap; reuses the vectors and "
                  << "the HNSW graph)." << std::endl;
      continue;
    }
    AD_LOG_INFO
        << "Loaded vector index '" << name << "' (" << idx.numLiveVectors()
        << " vectors"
        << (idx.numVectors() != idx.numLiveVectors()
                ? " + " +
                      std::to_string(idx.numVectors() - idx.numLiveVectors()) +
                      " tombstones"
                : "")
        << ", dim " << idx.dimensions() << ", scalar "
        << toString(idx.metadata().config_.scalar_)
        << (idx.hasRerankLayer()
                ? " + rerank " +
                      toString(idx.metadata().config_.rerankScalar_.value())
                : "")
        << ", HNSW=" << (idx.hasHnsw() ? "yes" : "no")
        << (idx.hasCsls()
                ? ", csls(k=" + std::to_string(idx.cslsNeighbors()) + ")"
                : "")
        << ")" << std::endl;
    // Apply the endpoint fields of the environment override BEFORE the
    // metadata fields are collected, so the `vec:model` metadata triple
    // reflects the effective model. (The "preload" field was already applied
    // at `open` above.)
    if (auto it = endpointOverrides.find(name); it != endpointOverrides.end()) {
      if (it->second.embeddingUrl_.has_value() ||
          it->second.embeddingModel_.has_value()) {
        idx.setEmbeddingEndpoint(std::move(it->second.embeddingUrl_),
                                 std::move(it->second.embeddingModel_));
        AD_LOG_INFO << "Vector index '" << name
                    << "': embedding endpoint overridden from "
                    << VECTOR_SEARCH_ENDPOINTS_ENV_VAR << std::endl;
      }
      // The fine-rerank batch size of the two-layer CSLS cut -- like the
      // endpoint fields, a pure in-memory serving setting reapplied on every
      // load.
      if (it->second.cslsRerankFloor_.has_value()) {
        idx.setCslsRerankFloor(it->second.cslsRerankFloor_.value());
        AD_LOG_INFO << "Vector index '" << name
                    << "': csls rerank floor set to "
                    << it->second.cslsRerankFloor_.value() << " at startup ("
                    << VECTOR_SEARCH_ENDPOINTS_ENV_VAR << ")" << std::endl;
      }
      endpointOverrides.erase(it);
    }
    const VectorIndexConfig& config = idx.metadata().config_;
    metadataFields.push_back({name, config.dimensions_,
                              toString(config.metric_),
                              toString(config.scalar_), idx.numLiveVectors(),
                              config.embeddingModel_});
    collection->add(name, std::move(idx));
    any = true;
  }
  // Warn about override names that matched no loaded index (a typo, or an
  // index that was skipped above), so a silently ineffective override is
  // visible in the log.
  for (const auto& entry : endpointOverrides) {
    AD_LOG_WARN << VECTOR_SEARCH_ENDPOINTS_ENV_VAR << " names vector index '"
                << entry.first
                << "', but no vector index of that name was loaded; the "
                   "override has no effect."
                << std::endl;
  }
  if (any) {
    logNumkongSimdOnce();
    impl.setExtension(std::string{VECTOR_EXTENSION_NAME},
                      std::move(collection));
    materializeMetadataTriples(impl, metadataFields);
  }
}

[[maybe_unused]] const bool registered = [] {
  IndexExtensionRegistry::get().addBuildHook(buildHook);
  IndexExtensionRegistry::get().addLoadHook(loadHook);
  return true;
}();

}  // namespace

// ____________________________________________________________________________
std::pair<uint64_t, uint64_t> remapVectorIndex(const Index& index,
                                               const std::string& basename,
                                               const std::string& name,
                                               uint32_t numThreads) {
  // 1. The existing index must be present and of the current format version.
  VectorIndexMetadata meta;
  {
    std::ifstream metaIn{vectorMetaFile(basename, name)};
    if (!metaIn.is_open()) {
      AD_THROW("Cannot remap vector index \"" + name +
               "\": its metadata file " + vectorMetaFile(basename, name) +
               " does not exist.");
    }
    nlohmann::json j;
    try {
      metaIn >> j;
      meta = j.get<VectorIndexMetadata>();
    } catch (const std::exception& e) {
      AD_THROW("Cannot remap vector index \"" + name +
               "\": its metadata is malformed: " + e.what());
    }
    // Remap only rewrites the small `.keys`/`.rowmap`/`.meta` files and never
    // touches `.data`/`.iris`, so it is safe for any SUPPORTED on-disk version
    // (a v4 index loads, so it must also be remappable). The rewritten metadata
    // is re-stamped at the current version below.
    if (!isSupportedVectorIndexVersion(meta.version_)) {
      AD_THROW("Cannot remap vector index \"" + name + "\": it has version " +
               std::to_string(meta.version_) +
               ", which this binary does not support. Please rebuild the "
               "vector index.");
    }
  }

  // 2. Re-resolve the row-aligned IRI sidecar against the new vocabulary
  //    (in parallel batches).
  std::ifstream irisIn{vectorIrisFile(basename, name)};
  if (!irisIn.is_open()) {
    AD_THROW("Cannot remap vector index \"" + name + "\": the IRI sidecar " +
             vectorIrisFile(basename, name) + " does not exist.");
  }
  const size_t threads = effectiveThreads(numThreads);
  std::vector<uint64_t> newKeys;
  newKeys.reserve(meta.numVectors_);
  std::vector<std::string> batch;
  std::string line;
  uint64_t lineNumber = 1;
  uint64_t tombstones = 0;
  auto flush = [&]() {
    auto ids = resolveBatch(index, batch, lineNumber, threads);
    for (const auto& id : ids) {
      if (id.has_value()) {
        newKeys.push_back(id.value().getBits());
      } else {
        newKeys.push_back(TOMBSTONE_KEY);
        ++tombstones;
      }
    }
    lineNumber += batch.size();
    batch.clear();
  };
  while (std::getline(irisIn, line)) {
    batch.push_back(line);
    if (batch.size() >= RESOLVE_BATCH_SIZE) {
      flush();
    }
  }
  flush();
  if (newKeys.size() != meta.numVectors_) {
    AD_THROW("Cannot remap vector index \"" + name +
             "\": the IRI sidecar has " + std::to_string(newKeys.size()) +
             " lines, but the index has " + std::to_string(meta.numVectors_) +
             " rows.");
  }

  // 3. Rewrite the two mapping files and the metadata (via temporaries; the
  //    metadata -- which carries the vocabulary fingerprint -- goes LAST, so
  //    an interrupted remap leaves a mismatching fingerprint and the index is
  //    skipped at load instead of serving a half-updated mapping).
  const std::string keysPath = vectorKeysFile(basename, name);
  const std::string rowmapPath = vectorRowmapFile(basename, name);
  const std::string metaPath = vectorMetaFile(basename, name);
  // Remove any `.tmp` outputs if we throw before the renames (e.g. disk full).
  RemapTmpCleanup cleanup{
      {keysPath + ".tmp", rowmapPath + ".tmp", metaPath + ".tmp"}};
  {
    ad_utility::MmapVector<uint64_t> keys;
    keys.open(newKeys.size(), keysPath + ".tmp");
    ql::ranges::copy(newKeys, keys.begin());
  }
  {
    std::vector<IdRowPair> pairs;
    pairs.reserve(newKeys.size() - tombstones);
    for (size_t row = 0; row < newKeys.size(); ++row) {
      if (newKeys[row] != TOMBSTONE_KEY) {
        pairs.push_back(IdRowPair{newKeys[row], row});
      }
    }
    ql::ranges::sort(pairs, std::less<>{},
                     [](const IdRowPair& pair) { return pair.idBits_; });
    ad_utility::MmapVector<IdRowPair> rowmap;
    rowmap.open(pairs.size(), rowmapPath + ".tmp");
    ql::ranges::copy(pairs, rowmap.begin());
  }
  meta.numTombstones_ = tombstones;
  meta.vocabSize_ = vocabFingerprint(index.getImpl());
  // Re-stamp the collation fingerprint against the (re-indexed) knowledge
  // graph: the `.data` layout is unchanged, but the mapping now refers to the
  // new vocabulary, so the guard should reflect its collation.
  meta.collationLocale_ = collationFingerprint(index.getImpl());
  {
    std::ofstream metaOut{metaPath + ".tmp"};
    if (!metaOut.is_open()) {
      AD_THROW("Could not write the vector metadata file " + metaPath + ".tmp");
    }
    metaOut << nlohmann::json(meta).dump(2);
    metaOut.close();
    if (metaOut.fail()) {
      AD_THROW("Could not write the vector metadata file " + metaPath +
               ".tmp (disk full?).");
    }
  }
  std::filesystem::rename(keysPath + ".tmp", keysPath);
  std::filesystem::rename(rowmapPath + ".tmp", rowmapPath);
  std::filesystem::rename(metaPath + ".tmp", metaPath);
  cleanup.dismiss();
  return {meta.numVectors_ - tombstones, tombstones};
}

}  // namespace qlever::vector
