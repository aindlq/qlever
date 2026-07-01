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
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>

#include "backports/StartsWithAndEndsWith.h"
#include "backports/algorithm.h"
#include "index/Index.h"
#include "index/IndexExtension.h"
#include "index/IndexImpl.h"
#include "parser/TripleComponent.h"
#include "rdfTypes/Iri.h"
#include "services/vectorSearch/EmbeddingClient.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorIndexFormat.h"
#include "services/vectorSearch/VectorInputReader.h"
#include "util/Exception.h"
#include "util/Log.h"
#include "util/json.h"

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

// The size of the loaded knowledge-graph vocabulary; used as the fingerprint
// that binds a vector index to the main-index build it was created from.
uint64_t vocabFingerprint(const IndexImpl& impl) {
  return static_cast<uint64_t>(impl.getVocab().size());
}

// Parse one `--service-index` JSON object entry into a build spec. Rejects
// unknown keys so that a typo does not silently fall back to a default.
VectorIndexBuildSpec parseSpec(const nlohmann::json& obj) {
  static constexpr std::array knownKeys{"name",
                                        "iris",
                                        "npy",
                                        "texts",
                                        "dimensions",
                                        "metric",
                                        "scalar",
                                        "hnsw",
                                        "hnswConnectivity",
                                        "hnswExpansionAdd",
                                        "hnswExpansionSearch",
                                        "embeddingUrl",
                                        "embeddingModel"};
  for (const auto& [key, value] : obj.items()) {
    if (ql::ranges::find(knownKeys, key) == knownKeys.end()) {
      AD_THROW("Unknown key \"" + key +
               "\" in a vectorSearch build spec. Known keys: name, iris, npy, "
               "texts, dimensions, metric, scalar, hnsw, hnswConnectivity, "
               "hnswExpansionAdd, hnswExpansionSearch, embeddingUrl, "
               "embeddingModel.");
    }
  }
  VectorIndexBuildSpec spec;
  try {
    spec.config_.name_ = obj.at("name").get<std::string>();
    spec.irisPath_ = obj.at("iris").get<std::string>();
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
    spec.config_.embeddingUrl_ = obj.value("embeddingUrl", std::string{});
    spec.config_.embeddingModel_ = obj.value("embeddingModel", std::string{});
  } catch (const nlohmann::json::exception& e) {
    AD_THROW(std::string{"Malformed vectorSearch build spec: "} + e.what());
  }
  if (spec.npyPath_.empty() == spec.textsPath_.empty()) {
    AD_THROW("Each vector index needs exactly one of `npy` or `texts`.");
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
  return spec;
}

// Resolve one line of the IRI input file to the entity's `Id`, or nullopt if
// the IRI is not part of the knowledge graph. Throws (with the line number) on
// lines that are not of the form `<...>`.
std::optional<Id> resolveIri(const Index& index, const std::string& iri,
                             uint64_t lineNumber) {
  if (iri.size() < 2 || !ql::starts_with(iri, "<") ||
      !ql::ends_with(iri, ">")) {
    AD_THROW("Line " + std::to_string(lineNumber) +
             " of the IRI input is not an IRI of the form <...>: \"" + iri +
             "\"");
  }
  TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(iri)};
  return tc.toValueId(index.getImpl());
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
  VectorIndexBuilder builder{basename, config};
  builder.setVocabSize(vocabFingerprint(index.getImpl()));
  std::string iri;
  std::vector<float> vec;
  uint64_t lineNumber = 0;
  while (reader.next(iri, vec)) {
    ++lineNumber;
    if (auto id = resolveIri(index, iri, lineNumber); id.has_value()) {
      builder.add(id.value(), vec);
      ++resolved;
    } else {
      ++skipped;
    }
  }
  return builder.build();
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
  std::optional<VectorIndexBuilder> builder;
  std::string iri;
  std::string text;
  uint64_t lineNumber = 0;
  while (std::getline(irisIn, iri)) {
    ++lineNumber;
    if (!std::getline(textsIn, text)) {
      AD_THROW("The texts file has fewer lines than the IRI list (" +
               std::to_string(lineNumber - 1) + " vs. at least " +
               std::to_string(lineNumber) + ").");
    }
    trim(iri);
    trim(text);
    auto id = resolveIri(index, iri, lineNumber);
    if (!id.has_value()) {
      ++skipped;
      continue;
    }
    std::vector<float> vec = embedTextOpenAI(
        config.embeddingUrl_, config.embeddingModel_, text, handle);
    if (!builder.has_value()) {
      if (config.dimensions_ == 0) {
        config.dimensions_ = static_cast<uint32_t>(vec.size());
      }
      builder.emplace(basename, config);
      builder->setVocabSize(vocabFingerprint(index.getImpl()));
    }
    builder->add(id.value(), vec);
    ++resolved;
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

// BUILD hook: build every vector index requested under the "vectorSearch" key.
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
    uint64_t resolved = 0;
    uint64_t skipped = 0;
    VectorIndexMetadata meta =
        spec.textsPath_.empty()
            ? buildFromNpy(index, basename, spec, resolved, skipped)
            : buildFromTexts(index, basename, spec, resolved, skipped);
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
      AD_LOG_WARN << "Skipping vector index '" << name
                  << "': it was built against a different version of the "
                     "knowledge graph (vocabulary size "
                  << idx.metadata().vocabSize_ << " at build time vs. "
                  << vocabFingerprint(impl)
                  << " now), so its entity ids are no longer valid. Rebuild it "
                     "with `--service-index`."
                  << std::endl;
      continue;
    }
    AD_LOG_INFO << "Loaded vector index '" << name << "' (" << idx.numVectors()
                << " vectors, dim " << idx.dimensions()
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
}  // namespace qlever::vector
