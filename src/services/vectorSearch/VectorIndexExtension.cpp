// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

// Self-registration of the vector index as a QLever *index extension*: a BUILD
// hook (run by `qlever index` after the main index is built, with URI->Id
// resolution available) and a LOAD hook (run at server start, after the vocab is
// loaded). This is the ONLY coupling to the index/build lifecycle, and it touches
// no core code -- everything lives in this folder. See `index/IndexExtension.h`.

#include "services/vectorSearch/VectorIndexExtension.h"

#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>

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

const VectorIndex* getVectorIndex(const Index& index, const std::string& name) {
  std::shared_ptr<void> ext =
      index.getImpl().getExtension(std::string{VECTOR_EXTENSION_NAME});
  if (!ext) {
    return nullptr;
  }
  return static_cast<const VectorIndexCollection*>(ext.get())->get(name);
}

namespace {

// Parse one `--service-index` JSON object entry into a build spec.
VectorIndexBuildSpec parseSpec(const nlohmann::json& obj) {
  VectorIndexBuildSpec spec;
  spec.config.name = obj.at("name").get<std::string>();
  spec.irisPath = obj.at("iris").get<std::string>();
  spec.npyPath = obj.value("npy", std::string{});
  spec.textsPath = obj.value("texts", std::string{});
  AD_CONTRACT_CHECK(spec.npyPath.empty() != spec.textsPath.empty(),
                    "Each vector index needs exactly one of `npy` or `texts`.");
  spec.config.dimensions = obj.value("dimensions", uint32_t{0});
  spec.config.metric =
      vectorMetricFromString(obj.value("metric", std::string{"cosine"}));
  spec.config.scalar =
      vectorScalarFromString(obj.value("scalar", std::string{"f32"}));
  spec.config.buildHnsw = obj.value("hnsw", true);
  spec.config.hnswConnectivity = obj.value("hnswConnectivity", uint32_t{16});
  spec.config.hnswExpansionAdd = obj.value("hnswExpansionAdd", uint32_t{128});
  spec.config.hnswExpansionSearch =
      obj.value("hnswExpansionSearch", uint32_t{64});
  spec.config.embeddingUrl = obj.value("embeddingUrl", std::string{});
  spec.config.embeddingModel = obj.value("embeddingModel", std::string{});
  return spec;
}

std::optional<Id> resolveIri(const Index& index, const std::string& iri) {
  TripleComponent tc{ad_utility::triple_component::Iri::fromIriref(iri)};
  return tc.toValueId(index.getImpl());
}

VectorIndexMetadata buildFromNpy(const Index& index, const std::string& basename,
                                 const VectorIndexBuildSpec& spec,
                                 uint64_t& resolved, uint64_t& skipped) {
  NpyVectorInputReader reader{spec.npyPath, spec.irisPath};
  VectorIndexConfig config = spec.config;
  if (config.dimensions == 0) {
    config.dimensions = reader.dimensions();
  }
  AD_CONTRACT_CHECK(config.dimensions == reader.dimensions(),
                    "Configured dimension does not match the .npy file.");
  VectorIndexBuilder builder{basename, config};
  std::string iri;
  std::vector<float> vec;
  while (reader.next(iri, vec)) {
    if (auto id = resolveIri(index, iri); id.has_value()) {
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
  VectorIndexConfig config = spec.config;
  AD_CONTRACT_CHECK(!config.embeddingUrl.empty(),
                    "Building a vector index from texts requires `embeddingUrl`.");
  std::ifstream irisIn{spec.irisPath};
  std::ifstream textsIn{spec.textsPath};
  AD_CONTRACT_CHECK(irisIn.is_open() && textsIn.is_open(),
                    "Could not open the iris/texts files.");
  auto handle =
      std::make_shared<ad_utility::SharedCancellationHandle::element_type>();
  std::optional<VectorIndexBuilder> builder;
  std::string iri;
  std::string text;
  while (std::getline(irisIn, iri) && std::getline(textsIn, text)) {
    while (!iri.empty() && (iri.back() == '\r' || iri.back() == ' ')) {
      iri.pop_back();
    }
    auto id = resolveIri(index, iri);
    if (!id.has_value()) {
      ++skipped;
      continue;
    }
    std::vector<float> vec =
        embedTextOpenAI(config.embeddingUrl, config.embeddingModel, text, handle);
    if (!builder.has_value()) {
      if (config.dimensions == 0) {
        config.dimensions = static_cast<uint32_t>(vec.size());
      }
      builder.emplace(basename, config);
    }
    builder->add(id.value(), vec);
    ++resolved;
  }
  AD_CONTRACT_CHECK(builder.has_value(),
                    "No input rows could be embedded/resolved.");
  return builder->build();
}

// BUILD hook: build every vector index requested under the "vectorSearch" key.
void buildHook(const Index& index, const std::string& basename,
               const nlohmann::json& fullConfig) {
  if (!fullConfig.contains(std::string{VECTOR_EXTENSION_NAME})) {
    return;
  }
  const auto& specs = fullConfig.at(std::string{VECTOR_EXTENSION_NAME});
  AD_CONTRACT_CHECK(specs.is_array(),
                    "The `vectorSearch` build config must be a JSON array.");
  for (const auto& obj : specs) {
    VectorIndexBuildSpec spec = parseSpec(obj);
    uint64_t resolved = 0;
    uint64_t skipped = 0;
    VectorIndexMetadata meta =
        spec.textsPath.empty()
            ? buildFromNpy(index, basename, spec, resolved, skipped)
            : buildFromTexts(index, basename, spec, resolved, skipped);
    AD_LOG_INFO << "Vector index '" << spec.config.name << "': indexed "
                << resolved << " vectors, skipped " << skipped
                << " (IRI not in the knowledge graph), HNSW="
                << (meta.hasHnsw ? "yes" : "no") << std::endl;
  }
}

// LOAD hook: memory-map every `<base>.vec.<name>.*` index and attach the
// collection to the `IndexImpl`.
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
        fn.compare(0, prefix.size(), prefix) != 0 ||
        fn.compare(fn.size() - suffix.size(), suffix.size(), suffix) != 0) {
      continue;
    }
    const std::string name =
        fn.substr(prefix.size(), fn.size() - prefix.size() - suffix.size());
    VectorIndex idx;
    idx.open(basename, name);
    AD_LOG_INFO << "Loaded vector index '" << name << "' (" << idx.numVectors()
                << " vectors, dim " << idx.dimensions()
                << ", HNSW=" << (idx.hasHnsw() ? "yes" : "no") << ")"
                << std::endl;
    collection->add(name, std::move(idx));
    any = true;
  }
  if (any) {
    impl.setExtension(std::string{VECTOR_EXTENSION_NAME}, std::move(collection));
  }
}

[[maybe_unused]] const bool registered = [] {
  IndexExtensionRegistry::get().addBuildHook(buildHook);
  IndexExtensionRegistry::get().addLoadHook(loadHook);
  return true;
}();

}  // namespace
}  // namespace qlever::vector
