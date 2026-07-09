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

// Build a tiny on-disk vector index. Endpoints and residency are serving
// concerns and are never persisted, so a freshly built index is always
// vector-only until an override is applied at load.
BuiltIndex buildTmpIndex() {
  BuiltIndex b;
  b.basename = uniqueTmpBasename();
  VectorIndexConfig cfg;
  cfg.name_ = b.name;
  cfg.dimensions_ = 4;
  cfg.metric_ = VectorMetric::Cosine;
  cfg.buildHnsw_ = false;
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
TEST(VectorIndexEndpointOverride, parseCslsRerankFloorOverride) {
  // A positive integer parses, alone or alongside other keys.
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"cslsRerankFloor": 20000},)"
      R"( "texts": {"embeddingModel": "m", "cslsRerankFloor": 1}})");
  ASSERT_EQ(overrides.size(), 2u);
  EXPECT_EQ(overrides.at("images").cslsRerankFloor_, 20000u);
  EXPECT_FALSE(overrides.at("images").embeddingUrl_.has_value());
  EXPECT_EQ(overrides.at("texts").cslsRerankFloor_, 1u);
  EXPECT_EQ(overrides.at("texts").embeddingModel_, "m");
  // Malformed values (zero, negative, fractional, string) poison the ENTIRE
  // entry -- even a valid field next to them must not half-apply -- while
  // well-formed entries in the same object still parse.
  auto invalid = parseEmbeddingEndpointOverrides(
      R"({"zero": {"cslsRerankFloor": 0},)"
      R"( "negative": {"cslsRerankFloor": -5},)"
      R"( "fractional": {"cslsRerankFloor": 1.5},)"
      R"( "string": {"embeddingModel": "m", "cslsRerankFloor": "10000"},)"
      R"( "good": {"cslsRerankFloor": 10}})");
  ASSERT_EQ(invalid.size(), 1u);
  EXPECT_EQ(invalid.at("good").cslsRerankFloor_, 10u);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, cslsRerankFloorOverrideIsInMemoryOnly) {
  auto b = buildTmpIndex();
  VectorIndex idx;
  idx.open(b.basename, b.name);
  EXPECT_EQ(idx.cslsRerankFloor(), DEFAULT_CSLS_RERANK_FLOOR);
  // Apply the override exactly as the load hook does.
  EndpointsEnvGuard guard{R"({"images": {"cslsRerankFloor": 123}})"};
  auto overrides = embeddingEndpointOverridesFromEnv();
  auto it = overrides.find(b.name);
  ASSERT_NE(it, overrides.end());
  ASSERT_TRUE(it->second.cslsRerankFloor_.has_value());
  idx.setCslsRerankFloor(it->second.cslsRerankFloor_.value());
  EXPECT_EQ(idx.cslsRerankFloor(), 123u);
  // The setter clamps to >= 1 (a floor of 0 would stall the rerank loop).
  idx.setCslsRerankFloor(0);
  EXPECT_EQ(idx.cslsRerankFloor(), 1u);
  // Never persisted: a fresh open is back at the default.
  VectorIndex reopened;
  reopened.open(b.basename, b.name);
  EXPECT_EQ(reopened.cslsRerankFloor(), DEFAULT_CSLS_RERANK_FLOOR);
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, parseAutoCutDefaultOverrides) {
  // The four per-index defaults of the dynamic `vec:autoCut` cuts parse,
  // alone or alongside other keys; `cslsFloor` may be negative.
  auto overrides = parseEmbeddingEndpointOverrides(
      R"({"images": {"cslsFloor": -0.25, "softmaxTemperature": 0.05,)"
      R"( "softmaxN": 42, "breadth": 0.75},)"
      R"( "texts": {"embeddingModel": "m", "breadth": 1}})");
  ASSERT_EQ(overrides.size(), 2u);
  EXPECT_FLOAT_EQ(overrides.at("images").cslsFloor_.value(), -0.25f);
  EXPECT_FLOAT_EQ(overrides.at("images").softmaxTemperature_.value(), 0.05f);
  EXPECT_EQ(overrides.at("images").softmaxN_, 42u);
  EXPECT_FLOAT_EQ(overrides.at("images").breadth_.value(), 0.75f);
  EXPECT_FALSE(overrides.at("images").cslsRerankFloor_.has_value());
  EXPECT_EQ(overrides.at("texts").embeddingModel_, "m");
  EXPECT_FLOAT_EQ(overrides.at("texts").breadth_.value(), 1.f);
  EXPECT_FALSE(overrides.at("texts").cslsFloor_.has_value());
  // Malformed values poison the ENTIRE entry (even a valid field next to
  // them must not half-apply), while well-formed entries still parse:
  // breadth out of [0, 1], non-positive/non-numeric temperature, softmaxN 0,
  // a string cslsFloor.
  auto invalid = parseEmbeddingEndpointOverrides(
      R"({"breadthHigh": {"embeddingModel": "m", "breadth": 1.5},)"
      R"( "breadthNeg": {"breadth": -0.1},)"
      R"( "tZero": {"softmaxTemperature": 0},)"
      R"( "tNeg": {"softmaxTemperature": -0.1},)"
      R"( "tString": {"softmaxTemperature": "0.1"},)"
      R"( "nZero": {"softmaxN": 0},)"
      R"( "nFrac": {"softmaxN": 1.5},)"
      R"( "floorString": {"cslsFloor": "low"},)"
      R"( "good": {"softmaxTemperature": 0.2}})");
  ASSERT_EQ(invalid.size(), 1u);
  EXPECT_FLOAT_EQ(invalid.at("good").softmaxTemperature_.value(), 0.2f);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, autoCutDefaultsAreInMemoryOnly) {
  auto b = buildTmpIndex();
  VectorIndex idx;
  idx.open(b.basename, b.name);
  // Fresh index: no serving defaults (the constants apply).
  EXPECT_FALSE(idx.cslsFloorDefault().has_value());
  EXPECT_FALSE(idx.softmaxTemperatureDefault().has_value());
  EXPECT_FALSE(idx.softmaxNDefault().has_value());
  EXPECT_FALSE(idx.breadthDefault().has_value());
  // Apply the overrides exactly as the load hook does.
  EndpointsEnvGuard guard{
      R"({"images": {"cslsFloor": -0.5, "softmaxTemperature": 0.2,)"
      R"( "softmaxN": 25, "breadth": 0.9}})"};
  auto overrides = embeddingEndpointOverridesFromEnv();
  auto it = overrides.find(b.name);
  ASSERT_NE(it, overrides.end());
  idx.setCslsFloorDefault(it->second.cslsFloor_);
  idx.setSoftmaxTemperatureDefault(it->second.softmaxTemperature_);
  idx.setSoftmaxNDefault(it->second.softmaxN_);
  idx.setBreadthDefault(it->second.breadth_);
  EXPECT_FLOAT_EQ(idx.cslsFloorDefault().value(), -0.5f);
  EXPECT_FLOAT_EQ(idx.softmaxTemperatureDefault().value(), 0.2f);
  EXPECT_EQ(idx.softmaxNDefault(), 25u);
  EXPECT_FLOAT_EQ(idx.breadthDefault().value(), 0.9f);
  // The setters clamp/ignore invalid values defensively.
  idx.setSoftmaxNDefault(0);
  EXPECT_EQ(idx.softmaxNDefault(), 1u);
  idx.setBreadthDefault(2.f);
  EXPECT_FLOAT_EQ(idx.breadthDefault().value(), 1.f);
  idx.setSoftmaxTemperatureDefault(-1.f);  // ignored, keeps 0.2
  EXPECT_FLOAT_EQ(idx.softmaxTemperatureDefault().value(), 0.2f);
  // `nullopt` resets to "use the constants".
  idx.setCslsFloorDefault(std::nullopt);
  EXPECT_FALSE(idx.cslsFloorDefault().has_value());
  // Never persisted: a fresh open has no defaults again.
  VectorIndex reopened;
  reopened.open(b.basename, b.name);
  EXPECT_FALSE(reopened.softmaxTemperatureDefault().has_value());
  EXPECT_FALSE(reopened.breadthDefault().has_value());
  cleanup(b);
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
  auto b = buildTmpIndex();
  const std::string metaPath = vectorMetaFile(b.basename, b.name);
  const std::string metaBefore = readFile(metaPath);

  VectorIndex idx;
  idx.open(b.basename, b.name);
  // The endpoint is never persisted, so a freshly built index is vector-only.
  EXPECT_TRUE(idx.metadata().config_.embeddingUrl_.empty());
  EXPECT_TRUE(idx.metadata().config_.embeddingModel_.empty());

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

  // The `.meta` is byte-identical (the override never reached the disk), and a
  // fresh open (a server start without the env var) is vector-only again.
  EXPECT_EQ(readFile(metaPath), metaBefore);
  VectorIndex reopened;
  reopened.open(b.basename, b.name);
  EXPECT_TRUE(reopened.metadata().config_.embeddingUrl_.empty());
  EXPECT_TRUE(reopened.metadata().config_.embeddingModel_.empty());
  cleanup(b);
}

// _____________________________________________________________________________
TEST(VectorIndexEndpointOverride, partialOverrideKeepsOtherField) {
  auto b = buildTmpIndex();
  VectorIndex idx;
  idx.open(b.basename, b.name);
  // Establish an in-memory baseline endpoint (as a full override would).
  idx.setEmbeddingEndpoint("http://build-time:8080", "clip");
  EXPECT_EQ(idx.metadata().config_.embeddingUrl_, "http://build-time:8080");
  EXPECT_EQ(idx.metadata().config_.embeddingModel_, "clip");

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
  // A vector-only index (the only kind the builder produces, since endpoints
  // are never persisted) can be given an endpoint at server start.
  auto b = buildTmpIndex();
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
  // With no override the residency is the default, "none" (mmap-only).
  auto b = buildTmpIndex();
  const std::string metaPath = vectorMetaFile(b.basename, b.name);
  const std::string metaBefore = readFile(metaPath);

  // Without an override the default wins: mmap-only.
  {
    VectorIndex idx;
    idx.open(b.basename, b.name);
    EXPECT_EQ(idx.metadata().config_.preload_, "none");
    EXPECT_EQ(idx.residency(), VectorIndex::Residency::None);
  }

  // Drive the exact load-hook path: env var -> parse -> map via
  // `residencyFromString` -> thread the residency into `open`. The index now
  // reports the requested residency; the `preload` config field (never
  // persisted, so always its default) still reads "none".
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
