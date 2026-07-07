// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORINDEXEXTENSION_H

#include <charconv>
#include <cstdint>
#include <cstdlib>
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

// The environment variable through which the embedding endpoints persisted in
// the per-index `.meta` files can be set or changed AT SERVER START, without
// rebuilding the index: a JSON object keyed by index name, each value an
// object with the optional string keys "embeddingUrl" and "embeddingModel",
// e.g.
//   QLEVER_VECTOR_SEARCH_ENDPOINTS='{
//     "images":   {"embeddingUrl": "unix:/siglip2.private",
//                  "embeddingModel": "siglip"},
//     "metadata": {"embeddingUrl": "unix:/qwen3.private"}}'
// Only the fields present are overridden. The load hook applies the override
// IN MEMORY on every server start; the on-disk `.meta` is never rewritten, so
// starting without the variable falls back to the persisted endpoints.
inline constexpr const char* VECTOR_SEARCH_ENDPOINTS_ENV_VAR =
    "QLEVER_VECTOR_SEARCH_ENDPOINTS";

// One per-index override parsed from the environment variable above; a
// `nullopt` field keeps the persisted value.
struct EmbeddingEndpointOverride {
  std::optional<std::string> embeddingUrl_;
  std::optional<std::string> embeddingModel_;
};
using EmbeddingEndpointOverrides =
    ad_utility::HashMap<std::string, EmbeddingEndpointOverride>;

// Parse the JSON value of `QLEVER_VECTOR_SEARCH_ENDPOINTS` into per-index
// overrides. NEVER throws -- a bad value must not prevent the server from
// starting: an empty value yields an empty map, malformed JSON (or a
// non-object top level) logs a warning and yields an empty map, and a
// malformed entry (non-object value, unknown key, non-string field, or no
// fields at all) logs a warning and is skipped ENTIRELY, so a typo cannot
// half-apply an override.
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
                  << "': expected an object with the optional string keys "
                     "\"embeddingUrl\" and \"embeddingModel\"."
                  << std::endl;
      continue;
    }
    EmbeddingEndpointOverride endpointOverride;
    bool ok = true;
    for (const auto& [key, field] : value.items()) {
      if (key == "embeddingUrl" && field.is_string()) {
        endpointOverride.embeddingUrl_ = field.get<std::string>();
      } else if (key == "embeddingModel" && field.is_string()) {
        endpointOverride.embeddingModel_ = field.get<std::string>();
      } else {
        AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                    << " entry for vector index '" << name << "': the key \""
                    << key
                    << "\" is unknown or not a string (known keys: "
                       "\"embeddingUrl\", \"embeddingModel\")."
                    << std::endl;
        ok = false;
        break;
      }
    }
    if (!endpointOverride.embeddingUrl_.has_value() &&
        !endpointOverride.embeddingModel_.has_value()) {
      if (ok) {
        AD_LOG_WARN << "Ignoring the " << VECTOR_SEARCH_ENDPOINTS_ENV_VAR
                    << " entry for vector index '" << name
                    << "': it overrides neither \"embeddingUrl\" nor "
                       "\"embeddingModel\"."
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
