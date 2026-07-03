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
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#include "backports/StartsWithAndEndsWith.h"
#include "backports/algorithm.h"
#include "index/Index.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "index/StringSortComparator.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "services/vectorSearch/EmbeddingClient.h"
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

// How many input rows are read, resolved, and (for the texts input) embedded
// per batch.
constexpr size_t RESOLVE_BATCH_SIZE = 16384;
constexpr size_t EMBED_BATCH_SIZE = 64;

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
                                        "texts",
                                        "parquet",
                                        "dimensions",
                                        "metric",
                                        "scalar",
                                        "hnsw",
                                        "hnswConnectivity",
                                        "hnswExpansionAdd",
                                        "hnswExpansionSearch",
                                        "buildThreads",
                                        "embeddingUrl",
                                        "embeddingModel",
                                        "alignRows",
                                        "preload",
                                        "remap"};
  for (const auto& [key, value] : obj.items()) {
    if (ql::ranges::find(knownKeys, key) == knownKeys.end()) {
      AD_THROW("Unknown key \"" + key +
               "\" in a vectorSearch build spec. Known keys: name, iris, npy, "
               "texts, parquet, dimensions, metric, scalar, hnsw, "
               "hnswConnectivity, hnswExpansionAdd, hnswExpansionSearch, "
               "buildThreads, embeddingUrl, embeddingModel, alignRows, "
               "preload, remap.");
    }
  }
  try {
    spec.parquetPath_ = obj.value("parquet", std::string{});
    // Parquet carries the URIs itself; the other inputs need the sidecar.
    spec.irisPath_ = spec.parquetPath_.empty()
                         ? obj.at("iris").get<std::string>()
                         : obj.value("iris", std::string{});
    spec.npyPath_ = obj.value("npy", std::string{});
    spec.textsPath_ = obj.value("texts", std::string{});
    spec.config_.dimensions_ = obj.value("dimensions", uint32_t{0});
    spec.config_.metric_ =
        vectorMetricFromString(obj.value("metric", std::string{"cosine"}));
    spec.config_.scalar_ =
        vectorScalarFromString(obj.value("scalar", std::string{"f32"}));
    spec.config_.buildHnsw_ = obj.value("hnsw", true);
    spec.config_.hnswConnectivity_ =
        obj.value("hnswConnectivity", uint32_t{16});
    spec.config_.hnswExpansionAdd_ =
        obj.value("hnswExpansionAdd", uint32_t{128});
    spec.config_.hnswExpansionSearch_ =
        obj.value("hnswExpansionSearch", uint32_t{64});
    spec.config_.buildThreads_ = obj.value("buildThreads", uint32_t{0});
    spec.config_.embeddingUrl_ = obj.value("embeddingUrl", std::string{});
    spec.config_.embeddingModel_ = obj.value("embeddingModel", std::string{});
    spec.config_.alignRows_ = obj.value("alignRows", false);
    spec.config_.preload_ = obj.value("preload", std::string{"none"});
  } catch (const nlohmann::json::exception& e) {
    AD_THROW(std::string{"Malformed vectorSearch build spec: "} + e.what());
  }
  // Validate the residency preference now (rather than at load time, on a
  // machine that may differ), so a typo is caught during `qlever index`.
  if (spec.config_.preload_ != "none" && spec.config_.preload_ != "advise" &&
      spec.config_.preload_ != "lock" && spec.config_.preload_ != "aligned") {
    AD_THROW(
        "`preload` must be one of \"none\", \"advise\", \"lock\", or "
        "\"aligned\" (got \"" +
        spec.config_.preload_ + "\").");
  }
  int numInputs = static_cast<int>(!spec.npyPath_.empty()) +
                  static_cast<int>(!spec.textsPath_.empty()) +
                  static_cast<int>(!spec.parquetPath_.empty());
  if (numInputs != 1) {
    AD_THROW(
        "Each vector index needs exactly one of `npy`, `texts`, or "
        "`parquet`.");
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

// Shared by the .npy and Parquet inputs: stream `(iri, vector)` rows from
// `reader` into a builder, resolving the IRIs in parallel batches.
VectorIndexMetadata buildFromReader(const Index& index,
                                    const std::string& basename,
                                    VectorIndexConfig config,
                                    VectorInputReader& reader,
                                    bool normalizeIris, uint64_t& resolved,
                                    uint64_t& skipped) {
  const size_t numThreads = effectiveThreads(config.buildThreads_);
  VectorIndexBuilder builder{basename, config};
  builder.setVocabSize(vocabFingerprint(index.getImpl()));
  builder.setCollationLocale(collationFingerprint(index.getImpl()));
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
      // Guard against a reader that reports a per-row length disagreeing with
      // the configured dimension (e.g. a Parquet `list` column whose actual
      // dimension the reader could not check up front) -- otherwise the row
      // would be silently mis-sliced into the wrong entity's vector.
      if (vec.size() != dim) {
        AD_THROW("Row " + std::to_string(lineNumber + iris.size()) +
                 " of the input has dimension " + std::to_string(vec.size()) +
                 ", but the index is configured with dimension " +
                 std::to_string(dim) + ".");
      }
      if (normalizeIris && !ql::starts_with(iri, "<")) {
        iri = "<" + iri + ">";
      }
      iris.push_back(iri);
      vectors.insert(vectors.end(), vec.begin(), vec.end());
    }
    auto ids = resolveBatch(index, iris, lineNumber, numThreads);
    for (size_t i = 0; i < iris.size(); ++i) {
      if (ids[i].has_value()) {
        builder.add(ids[i].value(), iris[i],
                    ql::span<const float>{vectors.data() + i * dim, dim});
        ++resolved;
      } else {
        ++skipped;
      }
    }
    lineNumber += iris.size();
  }
  return builder.build();
}

#ifdef QLEVER_WITH_PARQUET
VectorIndexMetadata buildFromParquet(const Index& index,
                                     const std::string& basename,
                                     const VectorIndexBuildSpec& spec,
                                     uint64_t& resolved, uint64_t& skipped) {
  ParquetVectorInputReader reader{spec.parquetPath_};
  VectorIndexConfig config = spec.config_;
  // For `list` columns the dimension is only known after the first row; peek
  // is not needed -- validate against the config where given, else infer.
  if (config.dimensions_ == 0 && reader.dimensions() == 0) {
    // Streamed inference: read the first row through a small detour.
    std::string iri;
    std::vector<float> vec;
    ParquetVectorInputReader probe{spec.parquetPath_};
    if (!probe.next(iri, vec)) {
      AD_THROW("The Parquet input " + spec.parquetPath_ + " is empty.");
    }
    config.dimensions_ = static_cast<uint32_t>(vec.size());
  } else if (config.dimensions_ == 0) {
    config.dimensions_ = reader.dimensions();
  } else if (reader.dimensions() != 0 &&
             config.dimensions_ != reader.dimensions()) {
    AD_THROW("The configured dimension (" + std::to_string(config.dimensions_) +
             ") does not match the Parquet file (" +
             std::to_string(reader.dimensions()) + ").");
  }
  return buildFromReader(index, basename, config, reader,
                         /*normalizeIris=*/true, resolved, skipped);
}
#endif  // QLEVER_WITH_PARQUET

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
  return buildFromReader(index, basename, config, reader,
                         /*normalizeIris=*/false, resolved, skipped);
}

VectorIndexMetadata buildFromTexts(const Index& index,
                                   const std::string& basename,
                                   const VectorIndexBuildSpec& spec,
                                   uint64_t& resolved, uint64_t& skipped) {
  VectorIndexConfig config = spec.config_;
  if (config.embeddingUrl_.empty()) {
    AD_THROW("Building a vector index from texts requires `embeddingUrl`.");
  }
  std::ifstream irisIn{spec.irisPath_};
  std::ifstream textsIn{spec.textsPath_};
  if (!irisIn.is_open()) {
    AD_THROW("Could not open the IRI list file " + spec.irisPath_);
  }
  if (!textsIn.is_open()) {
    AD_THROW("Could not open the texts file " + spec.textsPath_);
  }
  auto handle =
      std::make_shared<ad_utility::SharedCancellationHandle::element_type>();
  auto trim = [](std::string& s) {
    while (!s.empty() &&
           (s.back() == '\r' || s.back() == ' ' || s.back() == '\t')) {
      s.pop_back();
    }
  };
  const size_t numThreads = effectiveThreads(config.buildThreads_);
  std::optional<VectorIndexBuilder> builder;
  std::string iri;
  std::string text;
  uint64_t lineNumber = 0;
  bool done = false;
  // Rows are embedded in batches: one request per row would make the index
  // build latency-bound on the embedding endpoint.
  std::vector<std::string> batchIris;
  std::vector<std::string> batchTexts;
  auto flushBatch = [&]() {
    if (batchIris.empty()) {
      return;
    }
    uint64_t firstLine = lineNumber - batchIris.size() + 1;
    auto ids = resolveBatch(index, batchIris, firstLine, numThreads);
    std::vector<std::string> toEmbed;
    std::vector<size_t> toEmbedIdx;
    for (size_t i = 0; i < batchIris.size(); ++i) {
      if (ids[i].has_value()) {
        toEmbed.push_back(batchTexts[i]);
        toEmbedIdx.push_back(i);
      } else {
        ++skipped;
      }
    }
    if (toEmbed.empty()) {
      batchIris.clear();
      batchTexts.clear();
      return;
    }
    std::vector<std::vector<float>> embeddings = embedBatchOpenAI(
        config.embeddingUrl_, config.embeddingModel_, toEmbed, handle);
    for (size_t j = 0; j < toEmbed.size(); ++j) {
      size_t i = toEmbedIdx[j];
      const std::vector<float>& vec = embeddings[j];
      if (!builder.has_value()) {
        if (config.dimensions_ == 0) {
          config.dimensions_ = static_cast<uint32_t>(vec.size());
        }
        builder.emplace(basename, config);
        builder->setVocabSize(vocabFingerprint(index.getImpl()));
        builder->setCollationLocale(collationFingerprint(index.getImpl()));
      }
      builder->add(ids[i].value(), batchIris[i], vec);
      ++resolved;
    }
    batchIris.clear();
    batchTexts.clear();
  };
  while (!done) {
    if (!std::getline(irisIn, iri)) {
      done = true;
    } else {
      ++lineNumber;
      if (!std::getline(textsIn, text)) {
        AD_THROW("The texts file has fewer lines than the IRI list (" +
                 std::to_string(lineNumber - 1) + " vs. at least " +
                 std::to_string(lineNumber) + ").");
      }
      trim(iri);
      trim(text);
      batchIris.push_back(iri);
      batchTexts.push_back(text);
    }
    if (done || batchIris.size() >= EMBED_BATCH_SIZE) {
      flushBatch();
    }
  }
  if (std::getline(textsIn, text)) {
    trim(text);
    if (!text.empty()) {
      AD_THROW("The texts file has more (non-empty) lines than the IRI list (" +
               std::to_string(lineNumber) + ").");
    }
  }
  if (!builder.has_value()) {
    AD_THROW("No input rows could be embedded/resolved for vector index \"" +
             config.name_ + "\".");
  }
  return builder->build();
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
    VectorIndexMetadata meta;
    if (!spec.parquetPath_.empty()) {
#ifdef QLEVER_WITH_PARQUET
      meta = buildFromParquet(index, basename, spec, resolved, skipped);
#else
      AD_THROW(
          "This QLever build has no Parquet support; rebuild with "
          "-DQLEVER_VECTOR_SEARCH_PARQUET=ON (requires Apache "
          "Arrow/Parquet), or use the `npy` input.");
#endif
    } else if (!spec.textsPath_.empty()) {
      meta = buildFromTexts(index, basename, spec, resolved, skipped);
    } else {
      meta = buildFromNpy(index, basename, spec, resolved, skipped);
    }
    AD_LOG_INFO << "Vector index '" << spec.config_.name_ << "': indexed "
                << resolved << " vectors, skipped " << skipped
                << " (IRI not in the knowledge graph), HNSW="
                << (meta.hasHnsw_ ? "yes" : "no") << std::endl;
  }
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
  bool any = false;
  for (const auto& entry : fs::directory_iterator(dir, ec)) {
    const std::string fn = entry.path().filename().string();
    if (fn.size() <= prefix.size() + suffix.size() ||
        !ql::starts_with(fn, prefix) || !ql::ends_with(fn, suffix)) {
      continue;
    }
    const std::string name =
        fn.substr(prefix.size(), fn.size() - prefix.size() - suffix.size());
    VectorIndex idx;
    try {
      idx.open(basename, name);
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
    AD_LOG_INFO << "Loaded vector index '" << name << "' ("
                << idx.numLiveVectors() << " vectors"
                << (idx.numVectors() != idx.numLiveVectors()
                        ? " + " +
                              std::to_string(idx.numVectors() -
                                             idx.numLiveVectors()) +
                              " tombstones"
                        : "")
                << ", dim " << idx.dimensions()
                << ", HNSW=" << (idx.hasHnsw() ? "yes" : "no") << ")"
                << std::endl;
    collection->add(name, std::move(idx));
    any = true;
  }
  if (any) {
    impl.setExtension(std::string{VECTOR_EXTENSION_NAME},
                      std::move(collection));
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
