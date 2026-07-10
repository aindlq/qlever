// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H

#include <optional>
#include <string>
#include <vector>

#include "rdfTypes/Variable.h"
#include "services/vectorSearch/VectorIndexFormat.h"

// Dependency-light configuration shared by the vector-search parser
// (`parsedQuery::VectorSearchQuery`) and the engine operation (`VectorSearch`).
// Kept free of engine includes so the parser can use it without a cycle
// (mirrors `SpatialJoinConfig.h`).
namespace qlever::vector {

// A standalone vector similarity search that produces a `(?result[, ?score])`
// top-k table. The query point is supplied either as an explicit vector or as a
// constant entity IRI whose vector is looked up in the index.
struct VectorSearchConfiguration {
  // Name of the vector index to search (must be loaded).
  std::string indexName_;

  // An image query point (embedded at query time). Only URLs (which the
  // embedding endpoint fetches) and base64 payloads are supported; reading
  // server-local files from query text would be an arbitrary-file-read
  // primitive for remote clients.
  enum class ImageKind { Url, Base64 };
  struct ImageQuery {
    ImageKind kind_;
    std::string value_;  // URL, or base64 (raw or data URI)
  };

  // The query point. Exactly one of these specifies it:
  std::optional<std::vector<float>> queryVector_;  // an explicit vector,
  std::optional<std::string> queryEntityIri_;      // a constant entity IRI,
  std::optional<std::string> queryText_;           // free text to embed, or
  std::optional<ImageQuery> queryImage_;           // an image to embed.

  // True iff one of the query-point fields above is set (FORM W / FORM P;
  // FORM E -- entity-to-entity -- has no query point).
  bool hasQueryPoint() const {
    return queryVector_.has_value() || queryEntityIri_.has_value() ||
           queryText_.has_value() || queryImage_.has_value();
  }

  // The `vec:candidates` (alias `vec:left`) variable, bound by the
  // SURROUNDING query. With a query point it restricts the search to the
  // bound set (FORM P, pre-filter); without one it is the entity-to-entity
  // input (FORM E). See `VectorSearchQuery::toVectorSearchConfiguration`.
  std::optional<Variable> leftVariable_;

  // The variable bound to each result entity.
  Variable resultVariable_{"?_qlever_internal_vec_result"};

  // Optional variable bound to the similarity distance of each result (the
  // FINE, reranked distance on a two-layer index).
  std::optional<Variable> scoreVariable_;

  // Optional variable bound to the COARSE (scan-layer, e.g. i8) distance of
  // each result, for comparison against the fine `scoreVariable_` (e.g.
  // `ABS(?d - ?dc)` = the quantization error). On a single-layer index the
  // two layers coincide, so it binds the same distance as `scoreVariable_`.
  // NOTE: on a `binary` scan layer the coarse distance is the integer HAMMING
  // distance (0..dim, differing sign bits) -- a proxy on a DIFFERENT scale
  // than the fine cosine distance, so `ABS(?d - ?dc)` is meaningless there.
  std::optional<Variable> coarseScoreVariable_;

  // Number of nearest neighbours to return.
  size_t k_ = 10;

  // FORM P annotate only (`vec:candidates` and `vec:result` the SAME variable,
  // query point present, `vec:k` omitted): score ALL bound candidates instead
  // of keeping a top-`k_` cut (`k_` is ignored on the bound path then).
  bool keepAllCandidates_ = false;

  // Two-layer indices only: how many candidates the coarse scan pass keeps
  // for the fine rerank pass (`vec:rerankK`). Unset = `defaultRerankK` of the
  // index's scan scalar (`max(10 * k, 100)`; the far coarser 1-bit `binary`
  // layer keeps `max(50 * k, 500)`). Ignored on a single-layer index.
  std::optional<size_t> rerankK_;

  // Optional upper bound on the distance of returned neighbours.
  std::optional<float> maxDistance_;

  // CSLS retrieval cut (`vec:cslsThreshold`, requires an index built with
  // `csls: true` and a query point): replace the hardcoded top-k with the
  // query-adaptive filter `2 * cos_sim(q, d) - r(q) - r(d) >= threshold`.
  // The search becomes a FULL scan on the fine layer (CSLS needs every
  // candidate's cosine; the coarse/HNSW layer is not used), and the result
  // cardinality is variable: every candidate that "stands out" past the
  // threshold survives. `vec:bindScore` stays the COSINE DISTANCE (CSLS is
  // the cut, not the score) and results still sort by it ascending.
  std::optional<float> cslsThreshold_;
  // Optional variable bound to each survivor's CSLS value (`vec:bindCsls`).
  std::optional<Variable> cslsVariable_;
  // Optional override of the index's persisted `cslsNeighbors` for the
  // query-side r(q) (`vec:cslsNeighbors`); unset = the index's build value.
  std::optional<size_t> cslsNeighbors_;
  // With `cslsThreshold_` or `autoCut_`: the `vec:k` value iff it was
  // EXPLICITLY given (a cap on the cut's survivors); unset = return ALL
  // survivors (the variable cardinality that is the point of the cut). The
  // non-cut paths keep using `k_` (with its default of 10).
  std::optional<size_t> cslsKCap_;

  // DYNAMIC coverage-oriented cut (`vec:autoCut`, mutually exclusive with the
  // fixed `vec:cslsThreshold`; like it, requires a query point and runs the
  // coarse-scan + fine-rerank machinery). Two orthogonal dials:
  //
  //   `vec:autoCut` = the COVERAGE mode (how wide a net to cast):
  //     * Precise  -- only clear standouts (a high z-bar).
  //     * Balanced -- the neutral default.
  //     * Broad    -- a wide net (a low z-bar).
  //     * Exact    -- precise's bar, but returns NOTHING when no candidate
  //                   clears it (the explicit no-match answer) instead of the
  //                   single best. The others always return at least one.
  //
  //   `vec:cutSignal` = the METHOD / score the cut operates on (default
  //   `Cosine`):
  //     * Cosine -- a scale-INVARIANT noise-floor z-cut on the raw cosine
  //                 similarity (the cross-modal default; needs no csls data).
  //     * Csls   -- the same z-cut on the CSLS value `2*cos - r(q) - r(d)`
  //                 (requires an index built with `csls: true`). `vec:bindCsls`
  //                 works.
  //     * Softmax -- the softmax-standout cut: softmax the top-`softmaxN` fine
  //                 cosines at temperature `softmaxTemperature`, keep the
  //                 standouts `p_i >= alpha(mode) / softmaxN`
  //                 (csls-independent; `vec:bindCsls` is rejected).
  //
  // Back-compat: `vec:autoCut "softmax"` maps to (Softmax signal, Balanced),
  // `vec:autoCut "csls"` to (Csls signal, Balanced). See `resolveCslsCut`
  // (VectorSearch.h) for the query-param -> per-index default -> constant
  // resolution, and `qlever::vector::CslsCut` (VectorIndex.h) for the result.
  enum class CoverageMode { Precise, Balanced, Broad, Exact };
  std::optional<CoverageMode> autoCut_;
  // The cut method/signal (`vec:cutSignal`); only meaningful with `autoCut_`.
  enum class CutSignal { Cosine, Softmax, Csls };
  CutSignal cutSignal_ = CutSignal::Cosine;
  // Softmax signal only: the temperature T of `p_i = softmax(cos_i / T)`
  // (`vec:softmaxTemperature`). Unset = per-index default, then 0.1.
  std::optional<float> softmaxTemperature_;
  // Softmax signal only: how many of the best (fine-cosine) candidates enter
  // the softmax (`vec:softmaxN`). Unset = per-index default, then
  // `5 * cslsNeighbors`.
  std::optional<size_t> softmaxN_;

  // True iff a CSLS-machinery cut is requested (the fixed tau or a dynamic
  // autoCut) -- they share all engine plumbing.
  bool hasCslsCut() const {
    return cslsThreshold_.has_value() || autoCut_.has_value();
  }
  // True iff the requested cut derives a CSLS value (needs the sidecar): the
  // fixed threshold, or `vec:cutSignal "csls"`. Cosine/softmax do not.
  bool cutNeedsCsls() const {
    return cslsThreshold_.has_value() ||
           (autoCut_.has_value() && cutSignal_ == CutSignal::Csls);
  }

  // Optional algorithm override: force exact or approximate search. If unset,
  // the index decides (HNSW if available, else exact).
  enum class Algorithm { Automatic, Exact, Hnsw };
  Algorithm algorithm_ = Algorithm::Automatic;
};

}  // namespace qlever::vector

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORSEARCHCONFIG_H
