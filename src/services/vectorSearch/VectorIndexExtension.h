// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H

#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#include "backports/StartsWithAndEndsWith.h"
#include "services/vectorSearch/VectorIndex.h"
#include "util/Exception.h"
#include "util/HashMap.h"
#include "util/Log.h"
#include "util/json.h"

class Index;

namespace qlever::vector {

// The name under which the loaded vector indices are stored on the `IndexImpl`
// via the generic extension mechanism (see `index/IndexExtension.h`).
inline constexpr std::string_view VECTOR_EXTENSION_NAME = "vectorSearch";

// The auto-materialized metadata triples that make every loaded vector index
// queryable as an RDF resource (see
// `docs/vector-index/index-payload-design.md`, section "idx: metadata
// triples"): the load hook inserts, once per index,
//   <.../vectorSearch/index/NAME>  vec:dimension  3 ;
//       vec:metric "cosine" ; vec:precision "f32" ; vec:count 4 ;
//       vec:model "clip" .          # only if an embedding model is configured
// as DELTA triples. The IRIs live in the public `vectorSearch` namespace (NOT
// in QLever's internal `builtin-functions/` namespace), so the triples behave
// like ordinary inserted data and are visible to plain SPARQL.
inline constexpr std::string_view VECTOR_METADATA_SUBJECT_PREFIX =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/index/";
inline constexpr std::string_view VECTOR_METADATA_DIMENSION_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/dimension>";
inline constexpr std::string_view VECTOR_METADATA_METRIC_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/metric>";
inline constexpr std::string_view VECTOR_METADATA_PRECISION_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/precision>";
inline constexpr std::string_view VECTOR_METADATA_COUNT_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/count>";
inline constexpr std::string_view VECTOR_METADATA_MODEL_IRI =
    "<https://qlever.cs.uni-freiburg.de/vectorSearch/model>";

// The datatype-IRI prefix of a TYPED query-vector literal
// `"f0,f1,..."^^<PREFIX + MODEL + "/" + PRECISION>` (e.g. `.../vec/clip/f32`)
// -- the form `vec:embed` and `vec:vector` return. The datatype carries the
// embedding space (model + storage precision) of the index the vector came
// from, so `vec:distance` can VALIDATE that a query vector is comparable with
// the index it is measured against (same model, same precision, and -- via the
// float count -- same dimension). That makes CROSS-INDEX distances safe:
// `vec:distance(vidx:a, ?x, vec:vector(vidx:b, ?y))` computes iff the two
// indices share a space and is UNDEF otherwise. NOTE: unlike the metadata IRIs
// above, this prefix is stored WITHOUT angle brackets, because it is compared
// against `Literal::getDatatype()`, which strips them.
inline constexpr std::string_view VEC_QUERY_DATATYPE_PREFIX =
    "https://qlever.cs.uni-freiburg.de/vectorSearch/vec/";

// Build the (bracketless) query-vector datatype IRI for an index's embedding
// `model` (may be empty for a vector-only index without a declared model) and
// storage `precision` (a `toString(VectorScalar)` value, e.g. "f32").
inline std::string vectorQueryDatatypeIri(std::string_view model,
                                          std::string_view precision) {
  std::string out;
  out.reserve(VEC_QUERY_DATATYPE_PREFIX.size() + model.size() + 1 +
              precision.size());
  out.append(VEC_QUERY_DATATYPE_PREFIX)
      .append(model)
      .append("/")
      .append(precision);
  return out;
}

// The inverse: parse a (bracketless) datatype IRI back into (model,
// precision). The model is everything between the prefix and the LAST `/`
// (models may themselves contain slashes, e.g. `org/model-name`; it is empty
// for `.../vec//f32`), the precision is the final segment. Returns `nullopt`
// if `iri` is not under `VEC_QUERY_DATATYPE_PREFIX` or has no `/` separating
// the two parts.
inline std::optional<std::pair<std::string, std::string>>
parseVectorQueryDatatypeIri(std::string_view iri) {
  if (!ql::starts_with(iri, VEC_QUERY_DATATYPE_PREFIX)) {
    return std::nullopt;
  }
  iri.remove_prefix(VEC_QUERY_DATATYPE_PREFIX.size());
  size_t lastSlash = iri.rfind('/');
  if (lastSlash == std::string_view::npos) {
    return std::nullopt;
  }
  return std::pair{std::string{iri.substr(0, lastSlash)},
                   std::string{iri.substr(lastSlash + 1)}};
}

// Serialize a vector as a comma-separated float-list string, each float in its
// shortest round-trippable form (`std::to_chars`) -- the lexical form of the
// typed query-vector literal above, and exactly what `vec:distance` parses.
inline std::string toFloatListString(ql::span<const float> vector) {
  std::string out;
  // ~15 characters per shortest-round-trip float incl. the comma.
  out.reserve(vector.size() * 15);
  char buffer[64];
  for (float value : vector) {
    auto [ptr, ec] = std::to_chars(buffer, buffer + sizeof(buffer), value);
    AD_CORRECTNESS_CHECK(ec == std::errc{});
    if (!out.empty()) {
      out.push_back(',');
    }
    out.append(buffer, ptr);
  }
  return out;
}

// The inverse of the metadata-subject construction: extract the index name
// from an index IRI `<.../vectorSearch/index/NAME>` (the full string
// representation, including the angle brackets), or `nullopt` if `iri` is not
// of that form. This is how the `vec:distance`/`vec:embed` SPARQL functions
// resolve their first argument -- the SAME IRI that carries the queryable
// metadata triples above identifies the index to search.
inline std::optional<std::string> indexNameFromMetadataIri(
    std::string_view iri) {
  if (!ql::starts_with(iri, VECTOR_METADATA_SUBJECT_PREFIX) ||
      !ql::ends_with(iri, ">")) {
    return std::nullopt;
  }
  iri.remove_prefix(VECTOR_METADATA_SUBJECT_PREFIX.size());
  iri.remove_suffix(1);
  if (iri.empty()) {
    return std::nullopt;
  }
  return std::string{iri};
}

// The environment variable through which the embedding endpoints, the RAM
// residency (`preload`/`preloadRerank`), and the CSLS rerank floor are
// configured AT SERVER START. These are serving concerns, not index data:
// they are NEVER set at build time and NEVER persisted in the per-index
// `.meta` files -- this variable is their only source, applied fresh in
// memory on every server start. It is a JSON object keyed by index name, each
// value an object with the optional keys "embeddingUrl", "embeddingModel",
// "preload", and "preloadRerank" (strings; the latter two each
// "none"|"advise"|"lock"|"aligned") plus "cslsRerankFloor" (a positive
// integer -- the fine-rerank batch size of the two-layer CSLS cut, see
// `VectorIndex::cslsRerankFloor()`) and the per-index DEFAULTS of the
// dynamic `vec:autoCut` cuts (numbers; used when the query does not override
// them, see `resolveCslsCut`): "cslsFloor" (finite), "softmaxTemperature"
// (positive finite), "softmaxN" (a positive integer), "breadth" (in [0, 1]).
// E.g.
//   QLEVER_VECTOR_SEARCH_ENDPOINTS='{
//     "images":   {"embeddingUrl": "unix:/siglip2.private",
//                  "embeddingModel": "siglip",
//                  "softmaxTemperature": 0.05, "breadth": 0.7},
//     "metadata": {"embeddingUrl": "unix:/qwen3.private",
//                  "preload": "lock", "preloadRerank": "none",
//                  "cslsRerankFloor": 20000, "cslsFloor": -0.1}}'
// Only the fields present are set. The load hook applies them IN MEMORY; the
// on-disk `.meta` is never touched. The endpoint fields and the rerank floor
// mutate the already-opened index; "preload" (the SCAN matrix) and
// "preloadRerank" (the fine RERANK matrix of a two-layer index, ignored
// otherwise) must be decided WHEN the index is opened, so the load hook
// threads them into `VectorIndex::open(..., residency, rerankResidency)` as
// the per-layer residencies to apply -- e.g. pin the small i8 scan matrix
// ("preload": "lock") while the bf16 rerank matrix stays demand-paged (the
// default "preloadRerank": "none").
inline constexpr const char* VECTOR_SEARCH_ENDPOINTS_ENV_VAR =
    "QLEVER_VECTOR_SEARCH_ENDPOINTS";

// One per-index override parsed from the environment variable above; an absent
// field leaves the corresponding default in place (empty endpoint, `None`
// residency, `DEFAULT_CSLS_RERANK_FLOOR`, the `DEFAULT_CSLS_*` autoCut
// constants). `preload_` (the scan matrix) and `preloadRerank_` (the rerank
// matrix) are each validated to be one of "none"|"advise"|"lock"|"aligned"
// and select the per-layer RAM residency at `open` time; `cslsRerankFloor_`
// and `softmaxN_` are validated to be positive integers, `cslsFloor_` to be
// finite, `softmaxTemperature_` to be positive and finite, and `breadth_` to
// be in [0, 1].
struct EmbeddingEndpointOverride {
  std::optional<std::string> embeddingUrl_;
  std::optional<std::string> embeddingModel_;
  std::optional<std::string> preload_;
  std::optional<std::string> preloadRerank_;
  std::optional<size_t> cslsRerankFloor_;
  // Per-index serving DEFAULTS of the dynamic `vec:autoCut` cuts.
  std::optional<float> cslsFloor_;
  std::optional<float> softmaxTemperature_;
  std::optional<size_t> softmaxN_;
  std::optional<float> breadth_;
  // Per-index defaults of the TOP-ANCHORED z-cut (`zcut*` keys): the per-mode
  // band widths, the exact no-match gate, and the floor-estimator fraction.
  std::optional<float> zcutDeltaPrecise_;
  std::optional<float> zcutDeltaBalanced_;
  std::optional<float> zcutDeltaBroad_;
  std::optional<float> zcutGateZ_;
  std::optional<float> zcutFloorFraction_;

  bool empty() const {
    return !embeddingUrl_.has_value() && !embeddingModel_.has_value() &&
           !preload_.has_value() && !preloadRerank_.has_value() &&
           !cslsRerankFloor_.has_value() && !cslsFloor_.has_value() &&
           !softmaxTemperature_.has_value() && !softmaxN_.has_value() &&
           !breadth_.has_value() && !zcutDeltaPrecise_.has_value() &&
           !zcutDeltaBalanced_.has_value() && !zcutDeltaBroad_.has_value() &&
           !zcutGateZ_.has_value() && !zcutFloorFraction_.has_value();
  }
};
using EmbeddingEndpointOverrides =
    ad_utility::HashMap<std::string, EmbeddingEndpointOverride>;

// Parse the JSON value of `QLEVER_VECTOR_SEARCH_ENDPOINTS` into per-index
// overrides. NEVER throws -- a bad value must not prevent the server from
// starting: an empty value yields an empty map, malformed JSON (or a
// non-object top level) logs a warning and yields an empty map, and a
// malformed entry (non-object value, unknown key, a wrongly-typed field, an
// invalid "preload"/"cslsRerankFloor" value, or no fields at all) logs a
// warning and is skipped ENTIRELY, so a typo cannot half-apply an override.
inline EmbeddingEndpointOverrides parseEmbeddingEndpointOverrides(
    std::string_view json) {
  EmbeddingEndpointOverrides overrides;
  if (json.empty()) {
    return overrides;
  }
  nlohmann::json parsed;
  try {
    parsed = nlohmann::json::parse(json);
  } catch (const std::exception& e) {
    AD_LOG_WARN << "Ignoring " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                << ": its value is not valid JSON: " << e.what() << std::endl;
    return overrides;
  }
  if (!parsed.is_object()) {
    AD_LOG_WARN << "Ignoring " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                << ": expected a JSON object keyed by vector index name."
                << std::endl;
    return overrides;
  }
  for (const auto& [name, value] : parsed.items()) {
    if (!value.is_object()) {
      AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                  << " entry for vector index '" << name
                  << "': expected an object with the optional keys "
                     "\"embeddingUrl\", \"embeddingModel\", \"preload\", "
                     "\"preloadRerank\", \"cslsRerankFloor\", \"cslsFloor\", "
                     "\"softmaxTemperature\", \"softmaxN\", and \"breadth\"."
                  << std::endl;
      continue;
    }
    EmbeddingEndpointOverride endpointOverride;
    bool ok = true;
    // Validate one residency value (shared by "preload" and "preloadRerank").
    auto validPreload = [&name](std::string_view key,
                                const std::string& preload) {
      if (preload == "none" || preload == "advise" || preload == "lock" ||
          preload == "aligned") {
        return true;
      }
      AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                  << " entry for vector index '" << name << "': \"" << key
                  << "\" must be one of \"none\", \"advise\", "
                     "\"lock\", or \"aligned\" (got \""
                  << preload << "\")." << std::endl;
      return false;
    };
    for (const auto& [key, field] : value.items()) {
      if (key == "embeddingUrl" && field.is_string()) {
        endpointOverride.embeddingUrl_ = field.get<std::string>();
      } else if (key == "embeddingModel" && field.is_string()) {
        endpointOverride.embeddingModel_ = field.get<std::string>();
      } else if ((key == "preload" || key == "preloadRerank") &&
                 field.is_string()) {
        std::string preload = field.get<std::string>();
        if (!validPreload(key, preload)) {
          ok = false;
          break;
        }
        (key == "preload" ? endpointOverride.preload_
                          : endpointOverride.preloadRerank_) =
            std::move(preload);
      } else if (key == "cslsRerankFloor" || key == "softmaxN") {
        // The fine-rerank batch size of the two-layer CSLS cut, and the
        // softmax pool-size default of `vec:autoCut "softmax"`; each must be
        // a positive JSON integer (0 would stall the rerank loop / make the
        // softmax empty).
        if (!field.is_number_unsigned() || field.get<uint64_t>() == 0) {
          AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                      << " entry for vector index '" << name << "': \"" << key
                      << "\" must be a positive integer (got " << field.dump()
                      << ")." << std::endl;
          ok = false;
          break;
        }
        (key == "cslsRerankFloor" ? endpointOverride.cslsRerankFloor_
                                  : endpointOverride.softmaxN_) =
            static_cast<size_t>(field.get<uint64_t>());
      } else if (key == "cslsFloor" || key == "softmaxTemperature" ||
                 key == "breadth") {
        // Per-index defaults of the dynamic `vec:autoCut` cuts: the knee
        // floor (any finite number), the softmax temperature (positive), and
        // the breadth dial (in [0, 1]).
        const double v = field.is_number()
                             ? field.get<double>()
                             : std::numeric_limits<double>::quiet_NaN();
        const float f = static_cast<float>(v);
        const bool valid = key == "cslsFloor" ? std::isfinite(f)
                           : key == "softmaxTemperature"
                               ? std::isfinite(f) && f > 0.f
                               : f >= 0.f && f <= 1.f;
        if (!valid) {
          AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                      << " entry for vector index '" << name << "': \"" << key
                      << "\" must be "
                      << (key == "cslsFloor" ? "a finite number"
                          : key == "softmaxTemperature"
                              ? "a positive finite number"
                              : "a number between 0 and 1")
                      << " (got " << field.dump() << ")." << std::endl;
          ok = false;
          break;
        }
        (key == "cslsFloor"            ? endpointOverride.cslsFloor_
         : key == "softmaxTemperature" ? endpointOverride.softmaxTemperature_
                                       : endpointOverride.breadth_) = f;
      } else if (key == "zcutDeltaPrecise" || key == "zcutDeltaBalanced" ||
                 key == "zcutDeltaBroad" || key == "zcutGateZ" ||
                 key == "zcutFloorFraction") {
        // The top-anchored z-cut knobs: the band widths and the gate must be
        // positive finite; the floor fraction must be in (0, 1].
        const double v = field.is_number()
                             ? field.get<double>()
                             : std::numeric_limits<double>::quiet_NaN();
        const float f = static_cast<float>(v);
        const bool isFraction = key == "zcutFloorFraction";
        const bool valid =
            isFraction ? (f > 0.f && f <= 1.f) : (std::isfinite(f) && f > 0.f);
        if (!valid) {
          AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                      << " entry for vector index '" << name << "': \"" << key
                      << "\" must be "
                      << (isFraction ? "a number in (0, 1]"
                                     : "a positive finite number")
                      << " (got " << field.dump() << ")." << std::endl;
          ok = false;
          break;
        }
        if (key == "zcutDeltaPrecise") {
          endpointOverride.zcutDeltaPrecise_ = f;
        } else if (key == "zcutDeltaBalanced") {
          endpointOverride.zcutDeltaBalanced_ = f;
        } else if (key == "zcutDeltaBroad") {
          endpointOverride.zcutDeltaBroad_ = f;
        } else if (key == "zcutGateZ") {
          endpointOverride.zcutGateZ_ = f;
        } else {
          endpointOverride.zcutFloorFraction_ = f;
        }
      } else {
        AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                    << " entry for vector index '" << name << "': the key \""
                    << key
                    << "\" is unknown or has the wrong type (known keys: "
                       "\"embeddingUrl\", \"embeddingModel\", \"preload\", "
                       "\"preloadRerank\" -- strings -- "
                       "\"cslsRerankFloor\" and \"softmaxN\" -- positive "
                       "integers -- and \"cslsFloor\", "
                       "\"softmaxTemperature\", \"breadth\" -- numbers)."
                    << std::endl;
        ok = false;
        break;
      }
    }
    if (endpointOverride.empty()) {
      if (ok) {
        AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                    << " entry for vector index '" << name
                    << "': it overrides none of \"embeddingUrl\", "
                       "\"embeddingModel\", \"preload\", \"preloadRerank\", "
                       "\"cslsRerankFloor\", \"cslsFloor\", "
                       "\"softmaxTemperature\", \"softmaxN\", and "
                       "\"breadth\"."
                    << std::endl;
      }
      continue;
    }
    if (ok) {
      overrides.insert_or_assign(name, std::move(endpointOverride));
    }
  }
  return overrides;
}

// Read and parse `QLEVER_VECTOR_SEARCH_ENDPOINTS` from the environment (a
// no-op empty map if it is unset or empty). The load hook and the unit tests
// share this exact code path.
inline EmbeddingEndpointOverrides embeddingEndpointOverridesFromEnv() {
  const char* value = std::getenv(VECTOR_SEARCH_ENDPOINTS_ENV_VAR);
  return parseEmbeddingEndpointOverrides(
      value == nullptr ? std::string_view{} : std::string_view{value});
}

// All vector indices of a database, keyed by name. This is the object stored as
// the "vectorSearch" index extension and retrieved at query time.
class VectorIndexCollection {
 public:
  void add(const std::string& name, VectorIndex index) {
    indices_.insert_or_assign(name, std::move(index));
  }
  const VectorIndex* get(const std::string& name) const {
    auto it = indices_.find(name);
    return it == indices_.end() ? nullptr : &it->second;
  }

 private:
  ad_utility::HashMap<std::string, VectorIndex> indices_;
};

// Convenience for operations: the loaded `VectorIndex` named `name`, or an
// empty pointer. The returned `shared_ptr` aliases the whole collection, so the
// index stays valid even if the extension is ever replaced while in use.
std::shared_ptr<const VectorIndex> getVectorIndex(const Index& index,
                                                  const std::string& name);

// The human-readable, space-separated names of the SIMD ISAs set in
// `capabilities` (a NumKong `nk_capability_t` bitmask, e.g. from
// `nk_capabilities_available()`; declared as the underlying `uint64_t` so this
// header stays free of NumKong includes). Returns "serial(scalar)" if no SIMD
// bit is set (or this binary was built without NumKong). Backs the one-time
// "Vector search SIMD" startup log of the load hook; declared here so the
// pure bitmask -> name mapping is unit-testable.
std::string numkongActiveIsaString(uint64_t capabilities);

// Recompute the entity mapping of the existing on-disk vector index `name`
// against the CURRENTLY LOADED knowledge graph: re-resolves the row-aligned
// `.iris` sidecar (in parallel; `numThreads` 0 = all cores) and rewrites
// `.keys`/`.rowmap`/`.meta`. The vector data and the HNSW graph are reused
// as-is, which makes this orders of magnitude cheaper than a rebuild after the
// RDF data was re-indexed. Entities that are no longer part of the knowledge
// graph become tombstones. Returns (live entities, tombstones). Also invoked
// via `--service-index '{"vectorSearch":[{"name":"...","remap":true}]}'`.
std::pair<uint64_t, uint64_t> remapVectorIndex(const Index& index,
                                               const std::string& basename,
                                               const std::string& name,
                                               uint32_t numThreads = 0);

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
