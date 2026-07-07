// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests for the `QLEVER_VECTOR_SEARCH_ENDPOINTS` runtime override of an
// index's persisted embedding endpoint and RAM residency (`preload`): the
// env-JSON parsing (`parseEmbeddingEndpointOverrides` /
// `embeddingEndpointOverridesFromEnv`, the exact code path of the load hook),
// the in-memory-only `VectorIndex::setEmbeddingEndpoint` mutator it drives,
// and the `preload` -> `VectorIndex::open(..., residency)` path.

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "global/Id.h"
#include "global/IndexTypes.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexBuilder.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorIndexFormat.h"

using namespace qlever::vector;

namespace {

// A per-test unique path prefix, so that concurrently running tests (ctest -j)
// never race on shared filenames.
std::string uniqueTmpBasename() {
  const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
  return (std::filesystem::temp_directory_path() /
          (std::string{"qlever-vecenvtest-"} + info->test_suite_name() + "-" +
           info->name()))
      .string();
}

// RAII guard that sets `QLEVER_VECTOR_SEARCH_ENDPOINTS` for the duration of a
// scope and always unsets it afterwards, so a failing assertion can never leak
// the variable into later tests.
class EndpointsEnvGuard {
 public:
  explicit EndpointsEnvGuard(const char* value) {
    ::setenv(VECTOR_SEARCH_ENDPOINTS_ENV_VAR, value, /*overwrite=*/1);
  }
  ~EndpointsEnvGuard() { ::unsetenv(VECTOR_SEARCH_ENDPOINTS_ENV_VAR); }
  EndpointsEnvGuard(const EndpointsEnvGuard&) = delete;
  EndpointsEnvGuard& operator=(const EndpointsEnvGuard&) = delete;
};

struct BuiltIndex {
  std::string basename;
  std::string name = "images";
};

// Build a tiny on-disk vector index whose persisted config carries the given
// embedding endpoint (either may be empty, as for the `npy`/`parquet` build
// sources).
BuiltIndex buildTmpIndex(std::string url, std::string model) {
  BuiltIndex b;
  b.basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = b.name;
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
  cfg.embeddingUrl_ = std::move(url);
  cfg.embeddingModel_ = std::move(model);
  VectorIndexBuilder builder{b.basename, cfg};
  std::vector<float> v{1.f, 0.f, 0.f, 0.f};
  builder.add(Id::makeFromVocabIndex(VocabIndex::make(42)), "<http://ex/42>",
              v);
  builder.build();
  return b;
}

void cleanup(const BuiltIndex& b) {
  for (auto* suffix :
       {".meta", ".keys", ".rowmap", ".data", ".iris", ".hnsw"}) {
    std::error_code ec;
    std::filesystem::remove(b.basename + ".vec." + b.name + suffix, ec);
  }
}

std::string readFile(const std::string& path) {
  std::ifstream in{path, std::ios::binary};
  EXPECT_TRUE(in.is_open()) << path;
  std::ostringstream out;
  out << in.rdbuf();
  return std::move(out).str();
}

}  // namespace

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, parseValidOverrides) {
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"embeddingUrl": "unix:/siglip2.private",)"
      R"( "embeddingModel": "siglip"},)"
      R"( "metadata": {"embeddingUrl": "unix:/qwen3.private"},)"
      R"( "texts": {"embeddingModel": "qwen3"}})");
  ASSERT_EQ(overrides.size(), 3u);
  EXPECT_EQ(overrides.at("images").embeddingUrl_, "unix:/siglip2.private");
  EXPECT_EQ(overrides.at("images").embeddingModel_, "siglip");
  EXPECT_EQ(overrides.at("metadata").embeddingUrl_, "unix:/qwen3.private");
  EXPECT_FALSE(overrides.at("metadata").embeddingModel_.has_value());
  EXPECT_FALSE(overrides.at("texts").embeddingUrl_.has_value());
  EXPECT_EQ(overrides.at("texts").embeddingModel_, "qwen3");
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, parsePreloadOverride) {
  // `preload` parses alone and alongside the endpoint fields; all four valid
  // values are accepted (including the -- ineffective, see the load hook --
  // explicit "none").
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"preload": "lock"},)"
      R"( "texts": {"embeddingUrl": "unix:/x.sock", "preload": "advise"},)"
      R"( "a": {"preload": "none"}, "b": {"preload": "aligned"}})");
  ASSERT_EQ(overrides.size(), 4u);
  EXPECT_EQ(overrides.at("images").preload_, "lock");
  EXPECT_FALSE(overrides.at("images").embeddingUrl_.has_value());
  EXPECT_FALSE(overrides.at("images").embeddingModel_.has_value());
  EXPECT_EQ(overrides.at("texts").preload_, "advise");
  EXPECT_EQ(overrides.at("texts").embeddingUrl_, "unix:/x.sock");
  EXPECT_EQ(overrides.at("a").preload_, "none");
  EXPECT_EQ(overrides.at("b").preload_, "aligned");
  // An endpoint-only entry has no preload override.
  auto endpointOnly =
      parseEmbeddingEndpointOverrides(R"({"images": {"embeddingModel": "m"}})");
  ASSERT_EQ(endpointOnly.size(), 1u);
  EXPECT_FALSE(endpointOnly.at("images").preload_.has_value());
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, parseInvalidPreloadSkipsWholeEntry) {
  // An invalid `preload` value poisons the ENTIRE entry -- even the valid
  // `embeddingUrl` next to it must not half-apply -- while well-formed
  // entries in the same object still parse. A non-string `preload` is
  // likewise a malformed entry.
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"embeddingUrl": "unix:/x.sock", "preload": "locked"},)"
      R"( "nonString": {"preload": 1},)"
      R"( "good": {"preload": "aligned"}})");
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("good").preload_, "aligned");
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, parseMalformedNeverThrowsAndYieldsNothing) {
  // Whole value malformed or of the wrong shape -> empty map, no exception.
  EXPECT_TRUE(parseEmbeddingEndpointOverrides("").empty());
  EXPECT_TRUE(parseEmbeddingEndpointOverrides("{not json").empty());
  EXPECT_TRUE(parseEmbeddingEndpointOverrides("42").empty());
  EXPECT_TRUE(parseEmbeddingEndpointOverrides(R"(["images"])").empty());
  // Malformed entries are skipped entirely (no half-applied override), while
  // well-formed entries in the same object still apply.
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"notAnObject": "unix:/x.sock",)"
      R"( "typoKey": {"embedingUrl": "unix:/x.sock"},)"
      R"( "typoAndGood": {"embeddingUrl": "unix:/x.sock", "unknown": 1},)"
      R"( "nonString": {"embeddingUrl": 5},)"
      R"( "empty": {},)"
      R"( "good": {"embeddingModel": "m"}})");
  ASSERT_EQ(overrides.size(), 1u);
  EXPECT_EQ(overrides.at("good").embeddingModel_, "m");
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, envVarRoundTrip) {
  {
    EndpointsEnvGuard guard{R"({"images": {"embeddingModel": "siglip"}})"};
    auto overrides = embeddingEndpointOverridesFromEnv();
    ASSERT_EQ(overrides.size(), 1u);
    EXPECT_FALSE(overrides.at("images").embeddingUrl_.has_value());
    EXPECT_EQ(overrides.at("images").embeddingModel_, "siglip");
  }
  // Unset (the guard removed it) and empty values are a no-op.
  EXPECT_TRUE(embeddingEndpointOverridesFromEnv().empty());
  {
    EndpointsEnvGuard guard{""};
    EXPECT_TRUE(embeddingEndpointOverridesFromEnv().empty());
  }
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, overrideIsInMemoryOnly) {
  auto b = buildTmpIndex("http://build-time:8080", "clip");
  const std::string metaPath = vectorMetaFile(b.basename, b.name);
  const std::string metaBefore = readFile(metaPath);

  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "http://build-time:8080");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "clip");

  // Apply the override exactly as the load hook does: read the env var, look
  // up the opened index's name, and mutate the in-memory config.
  EndpointsEnvGuard guard{
      R"({"images": {"embeddingUrl": "unix:/siglip2.private",)"
      R"( "embeddingModel": "siglip"}})"};
  auto overrides = embeddingEndpointOverridesFromEnv();
  auto it = overrides.find(b.name);
  ASSERT_NE(it, overrides.end());
  idx.setEmbeddingEndpoint(std::move(it->second.embeddingUrl_),
                           std::move(it->second.embeddingModel_));
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "unix:/siglip2.private");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "siglip");

  // The persisted `.meta` is byte-identical, and a fresh open (a server start
  // without the env var) still reports the build-time endpoint: the override
  // never reached the disk.
  EXPECT_EQ(readFile(metaPath), metaBefore);
  VectorIndex reopened;
  reopened.open(b.basename, b.name);
  EXPECT_EQ(reopened.metadata().config_.embeddingUrl_,
            "http://build-time:8080");
  EXPECT_EQ(reopened.metadata().config_.embeddingModel_, "clip");
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, partialOverrideKeepsOtherField) {
  auto b = buildTmpIndex("http://build-time:8080", "clip");
  VectorIndex idx;
  idx.open(b.basename, b.name);

  // URL only: the model stays.
  idx.setEmbeddingEndpoint("unix:/new.sock", std::nullopt);
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "unix:/new.sock");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "clip");

  // Model only: the (already overridden) URL stays.
  idx.setEmbeddingEndpoint(std::nullopt, "siglip");
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "unix:/new.sock");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "siglip");

  // Neither: a no-op.
  idx.setEmbeddingEndpoint(std::nullopt, std::nullopt);
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "unix:/new.sock");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "siglip");
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, overrideSetsEndpointOnVectorOnlyIndex) {
  // An index built WITHOUT an endpoint (the optional `npy`/`parquet` case)
  // can be given one at server start.
  auto b = buildTmpIndex("", "");
  const std::string metaPath = vectorMetaFile(b.basename, b.name);
  const std::string metaBefore = readFile(metaPath);

  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_TRUE(idx.metadata().config_.embeddingUrl_.empty());
  EXPECT_TRUE(idx.metadata().config_.embeddingModel_.empty());

  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"embeddingUrl": "unix:/qwen3.private",)"
      R"( "embeddingModel": "qwen3"}})");
  auto it = overrides.find(b.name);
  ASSERT_NE(it, overrides.end());
  idx.setEmbeddingEndpoint(std::move(it->second.embeddingUrl_),
                           std::move(it->second.embeddingModel_));
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "unix:/qwen3.private");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "qwen3");
  EXPECT_EQ(readFile(metaPath), metaBefore);
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, residencyFromString) {
  using R = VectorIndex::Residency;
  EXPECT_EQ(VectorIndex::residencyFromString("none"), R::None);
  EXPECT_EQ(VectorIndex::residencyFromString("advise"), R::Advise);
  EXPECT_EQ(VectorIndex::residencyFromString("lock"), R::Lock);
  EXPECT_EQ(VectorIndex::residencyFromString("aligned"), R::AlignedCopy);
  // Unknown values (never produced by the validated parsers) fall back to
  // `None`, i.e. "use the persisted `preload`".
  EXPECT_EQ(VectorIndex::residencyFromString("bogus"), R::None);
  EXPECT_EQ(VectorIndex::residencyFromString(""), R::None);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, preloadOverrideAppliesResidencyAtOpen) {
  // The persisted `preload` of this index is the builder default, "none".
  auto b = buildTmpIndex("", "");
  const std::string metaPath = vectorMetaFile(b.basename, b.name);
  const std::string metaBefore = readFile(metaPath);

  // Without an override the persisted value wins: mmap-only.
  {
    VectorIndex idx;
    idx.open(b.basename, b.name);
    EXPECT_EQ(idx.metadata().config_.preload_, "none");
    EXPECT_EQ(idx.residency(), VectorIndex::Residency::None);
  }

  // Drive the exact load-hook path: env var -> parse -> map via
  // `residencyFromString` -> thread the residency into `open`. The index now
  // reports the overriding residency although its persisted `preload` (both
  // on disk and in the loaded metadata) still says "none".
  EndpointsEnvGuard guard{R"({"images": {"preload": "lock"}})"};
  auto overrides = embeddingEndpointOverridesFromEnv();
  auto it = overrides.find(b.name);
  ASSERT_NE(it, overrides.end());
  ASSERT_TRUE(it->second.preload_.has_value());
  VectorIndex idx;
  idx.open(b.basename, b.name,
           VectorIndex::residencyFromString(it->second.preload_.value()));
  EXPECT_EQ(idx.residency(), VectorIndex::Residency::Lock);
  EXPECT_EQ(idx.metadata().config_.preload_, "none");
  EXPECT_EQ(readFile(metaPath), metaBefore);
  cleanup(b);
}
