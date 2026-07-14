// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorSearchQuery.h"

#include <absl/strings/charconv.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <cmath>

#include "parser/SparqlTriple.h"
#include "services/vectorSearch/VectorIndexExtension.h"

namespace parsedQuery {

namespace {
// Parse a comma-separated list of floats, e.g. "0.1,-0.2,0.3". Throws a
// `VectorSearchException` on anything that is not a float.
std::vector<float> parseFloatList(std::string_view s) {
  std::vector<float> out;
  for (std::string_view token :
       absl::StrSplit(s, ',', absl::SkipWhitespace{})) {
    while (!token.empty() && (token.front() == ' ' || token.front() == '\t')) {
      token.remove_prefix(1);
    }
    while (!token.empty() && (token.back() == ' ' || token.back() == '\t')) {
      token.remove_suffix(1);
    }
    float value{};
    auto [ptr, ec] =
        absl::from_chars(token.data(), token.data() + token.size(), value);
    if (ec != std::errc{} || ptr != token.data() + token.size() ||
        !std::isfinite(value)) {
      throw VectorSearchException{absl::StrCat(
          "`<queryVector>` contains a value that is not a finite number: \"",
          token, "\".")};
    }
    out.push_back(value);
  }
  return out;
}
}  // namespace

// ____________________________________________________________________________
void VectorSearchQuery::addParameter(const SparqlTriple& triple) {
  auto simpleTriple = triple.getSimple();
  TripleComponent predicate = simpleTriple.p_;
  TripleComponent object = simpleTriple.o_;
  auto pred = extractParameterName(predicate, VECTOR_SEARCH_IRI);

  auto requireLiteral = [&object, &pred]() -> std::string {
    if (!object.isLiteral()) {
      throw VectorSearchException{absl::StrCat("The parameter `<", pred,
                                               ">` expects a string literal.")};
    }
    return std::string{asStringViewUnsafe(object.getLiteral().getContent())};
  };

  if (pred == "index") {
    // `<index>` accepts either a plain string literal ("images") or the
    // index's metadata IRI (`<.../vectorSearch/index/images>`) -- the same
    // IRI the `vec:distance`/`vec:embed` functions and `vec:hasMember` take.
    if (object.isLiteral()) {
      indexName_ =
          std::string{asStringViewUnsafe(object.getLiteral().getContent())};
    } else if (object.isIri()) {
      std::optional<std::string> name =
          qlever::vector::indexNameFromMetadataIri(
              object.getIri().toStringRepresentation());
      if (!name.has_value()) {
        throw VectorSearchException{absl::StrCat(
            "The parameter `<index>` expects a string literal (e.g. "
            "\"images\") or a vector-index IRI "
            "`<https://qlever.cs.uni-freiburg.de/vectorSearch/index/NAME>`, "
            "but got the IRI ",
            object.getIri().toStringRepresentation(), ".")};
      }
      indexName_ = std::move(name).value();
    } else {
      throw VectorSearchException{
          "The parameter `<index>` expects a string literal (e.g. \"images\") "
          "or a vector-index IRI "
          "`<https://qlever.cs.uni-freiburg.de/vectorSearch/index/NAME>`."};
    }
  } else if (pred == "query") {
    if (!object.isIri()) {
      throw VectorSearchException{
          "The parameter `<query>` expects an entity IRI whose vector is used "
          "as the query point."};
    }
    queryEntityIri_ = object.getIri().toStringRepresentation();
  } else if (pred == "queryVector") {
    queryVector_ = parseFloatList(requireLiteral());
  } else if (pred == "queryText") {
    queryText_ = requireLiteral();
  } else if (pred == "imageUrl") {
    using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
    // Use the bare IRI content (`https://...`), NOT `toStringRepresentation()`
    // (`<https://...>`): the raw URL is forwarded to the embedding endpoint,
    // which rejects the angle brackets. Matches how `vec:embed` reads its
    // image IRI (`getContent()`).
    std::string url =
        object.isIri()
            ? std::string{asStringViewUnsafe(object.getIri().getContent())}
            : requireLiteral();
    queryImage_ = {ImageKind::Url, std::move(url)};
  } else if (pred == "imageBase64") {
    using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
    queryImage_ = {ImageKind::Base64, requireLiteral()};
  } else if (pred == "candidates" || pred == "left") {
    // `<candidates>` is the canonical name; `<left>` is the original one,
    // kept as a working alias.
    setVariable("candidates", object, leftVar_);
  } else if (pred == "result" || pred == "right") {
    setVariable("result", object, resultVar_);
  } else if (pred == "bindScore") {
    setVariable("bindScore", object, scoreVar_);
  } else if (pred == "bindCoarseScore") {
    setVariable("bindCoarseScore", object, coarseScoreVar_);
  } else if (pred == "k" || pred == "numNearestNeighbors") {
    if (!object.isInt() || object.getInt() <= 0) {
      throw VectorSearchException{
          "The parameter `<k>` expects a positive integer."};
    }
    k_ = static_cast<size_t>(object.getInt());
  } else if (pred == "rerankK") {
    if (!object.isInt() || object.getInt() <= 0) {
      throw VectorSearchException{
          "The parameter `<rerankK>` expects a positive integer (the number "
          "of coarse-scan candidates kept for the rerank pass of a two-layer "
          "index)."};
    }
    rerankK_ = static_cast<size_t>(object.getInt());
  } else if (pred == "maxDistance") {
    std::optional<double> value;
    if (object.isInt()) {
      value = static_cast<double>(object.getInt());
    } else if (object.isDouble()) {
      value = object.getDouble();
    }
    if (!value.has_value()) {
      throw VectorSearchException{
          "The parameter `<maxDistance>` expects a number (the maximum "
          "distance of returned neighbours)."};
    }
    maxDistance_ = static_cast<float>(value.value());
  } else if (pred == "cslsThreshold") {
    std::optional<double> value;
    if (object.isInt()) {
      value = static_cast<double>(object.getInt());
    } else if (object.isDouble()) {
      value = object.getDouble();
    }
    if (!value.has_value() || !std::isfinite(value.value())) {
      throw VectorSearchException{
          "The parameter `<cslsThreshold>` expects a finite number (the CSLS "
          "cut: candidates with `2*cos_sim - r(q) - r(d)` below it are "
          "dropped; 0 is a balanced starting point, negative values keep "
          "more)."};
    }
    cslsThreshold_ = static_cast<float>(value.value());
  } else if (pred == "bindCsls") {
    setVariable("bindCsls", object, cslsVar_);
  } else if (pred == "autoCut") {
    using CoverageMode =
        qlever::vector::VectorSearchConfiguration::CoverageMode;
    using CutSignal = qlever::vector::VectorSearchConfiguration::CutSignal;
    // A back-compat `vec:autoCut "softmax"/"csls"` fixes the signal too; a
    // conflicting explicit `vec:cutSignal` is rejected below.
    auto setSignalOnce = [this](CutSignal s) {
      if (cutSignal_.has_value() && cutSignal_.value() != s) {
        throw VectorSearchException{
            "Conflicting `<cutSignal>` and back-compat `<autoCut> "
            "\"softmax\"/\"csls\"`; use the coverage words for `<autoCut>` and "
            "set the signal with `<cutSignal>`."};
      }
      cutSignal_ = s;
    };
    std::string mode = requireLiteral();
    if (mode == "precise") {
      autoCut_ = CoverageMode::Precise;
    } else if (mode == "balanced") {
      autoCut_ = CoverageMode::Balanced;
    } else if (mode == "broad") {
      autoCut_ = CoverageMode::Broad;
    } else if (mode == "exact") {
      autoCut_ = CoverageMode::Exact;
    } else if (mode == "softmax") {
      autoCut_ = CoverageMode::Balanced;
      setSignalOnce(CutSignal::Softmax);
    } else if (mode == "csls") {
      autoCut_ = CoverageMode::Balanced;
      setSignalOnce(CutSignal::Csls);
    } else {
      throw VectorSearchException{absl::StrCat(
          "The parameter `<autoCut>` expects a coverage mode -- \"precise\", "
          "\"balanced\", \"broad\", or \"exact\" -- (or the back-compat "
          "\"softmax\"/\"csls\" signal shortcuts), but got \"",
          mode, "\".")};
    }
  } else if (pred == "cutSignal") {
    using CutSignal = qlever::vector::VectorSearchConfiguration::CutSignal;
    std::string sig = requireLiteral();
    CutSignal s;
    if (sig == "cosine") {
      s = CutSignal::Cosine;
    } else if (sig == "softmax") {
      s = CutSignal::Softmax;
    } else if (sig == "csls") {
      s = CutSignal::Csls;
    } else {
      throw VectorSearchException{absl::StrCat(
          "The parameter `<cutSignal>` expects \"cosine\" (the scale-invariant "
          "noise-floor cut, the default), \"softmax\", or \"csls\", but got "
          "\"",
          sig, "\".")};
    }
    if (cutSignal_.has_value() && cutSignal_.value() != s) {
      throw VectorSearchException{
          "Conflicting `<cutSignal>` and back-compat `<autoCut> "
          "\"softmax\"/\"csls\"`; use the coverage words for `<autoCut>` and "
          "set the signal with `<cutSignal>`."};
    }
    cutSignal_ = s;
  } else if (pred == "softmaxTemperature") {
    std::optional<double> value;
    if (object.isInt()) {
      value = static_cast<double>(object.getInt());
    } else if (object.isDouble()) {
      value = object.getDouble();
    }
    if (!value.has_value() || !std::isfinite(value.value()) ||
        value.value() <= 0.0) {
      throw VectorSearchException{
          "The parameter `<softmaxTemperature>` expects a positive finite "
          "number (the T of the `<cutSignal> \"softmax\"` cut's "
          "`softmax(cos / T)`; default 0.1, smaller = peakier)."};
    }
    softmaxTemperature_ = static_cast<float>(value.value());
  } else if (pred == "softmaxN") {
    if (!object.isInt() || object.getInt() <= 0) {
      throw VectorSearchException{
          "The parameter `<softmaxN>` expects a positive integer (how many "
          "of the best candidates enter the `<cutSignal> \"softmax\"` cut; "
          "default `5 * cslsNeighbors`)."};
    }
    softmaxN_ = static_cast<size_t>(object.getInt());
  } else if (pred == "cslsNeighbors") {
    if (!object.isInt() || object.getInt() <= 0) {
      throw VectorSearchException{
          "The parameter `<cslsNeighbors>` expects a positive integer (the "
          "neighbour count of the query-side r(q); default: the index's "
          "build-time `cslsNeighbors`)."};
    }
    cslsNeighbors_ = static_cast<size_t>(object.getInt());
  } else if (pred == "algorithm") {
    if (!object.isIri()) {
      throw VectorSearchException{
          "The parameter `<algorithm>` expects `vectorSearch:exact`, "
          "`vectorSearch:hnsw`, or `vectorSearch:auto`."};
    }
    using Algo = qlever::vector::VectorSearchConfiguration::Algorithm;
    auto a = extractParameterName(object, VECTOR_SEARCH_IRI);
    if (a == "exact") {
      algo_ = Algo::Exact;
    } else if (a == "hnsw") {
      algo_ = Algo::Hnsw;
    } else if (a == "auto") {
      algo_ = Algo::Automatic;
    } else {
      throw VectorSearchException{
          absl::StrCat("Unknown vector-search algorithm `", a, "`.")};
    }
  } else if (pred == "fullPrecision") {
    // Accept a bare xsd:boolean (`vec:fullPrecision true`) or a string
    // (`vec:fullPrecision "true"`).
    if (object.isBool()) {
      fullPrecision_ = object.getBool();
    } else if (object.isLiteral()) {
      std::string v{asStringViewUnsafe(object.getLiteral().getContent())};
      if (v == "true" || v == "1") {
        fullPrecision_ = true;
      } else if (v == "false" || v == "0") {
        fullPrecision_ = false;
      } else {
        throw VectorSearchException{
            "The parameter `<fullPrecision>` expects a boolean (`true`/`false`, "
            "bare or as a string)."};
      }
    } else {
      throw VectorSearchException{
          "The parameter `<fullPrecision>` expects a boolean (`true`/`false`). "
          "When true, the search skips the quantized coarse layer and scans "
          "the full-precision fine layer (e.g. bf16) exhaustively -- no coarse "
          "preselection or rerank."};
    }
  } else if (pred == "i8Kernel") {
    // Which exact-i8-cosine kernel the coarse scan layer uses -- a
    // performance A/B dial (identical results on a VNNI CPU: both engines
    // share one integer-dot + finalize). A string literal
    // (`"vnni"`/`"auto"`/`"punned"`); unset = `Auto`.
    if (!object.isLiteral()) {
      throw VectorSearchException{
          "The parameter `<i8Kernel>` expects a string "
          "(`\"vnni\"`, `\"auto\"`, or `\"punned\"`)."};
    }
    std::string v{asStringViewUnsafe(object.getLiteral().getContent())};
    if (v == "auto") {
      i8Kernel_ = qlever::vector::I8Kernel::Auto;
    } else if (v == "vnni") {
      i8Kernel_ = qlever::vector::I8Kernel::Vnni;
    } else if (v == "punned") {
      i8Kernel_ = qlever::vector::I8Kernel::Punned;
    } else {
      throw VectorSearchException{absl::StrCat(
          "Unknown `<i8Kernel>` value `", v,
          "`; expected `\"vnni\"`, `\"auto\"`, or `\"punned\"`. This is a "
          "performance A/B dial (identical results across kernels); a kernel "
          "the CPU cannot run is silently downgraded.")};
    }
  } else if (pred == "bf16Kernel") {
    // Which exact-bf16-cosine kernel the fine layer uses -- a performance A/B
    // dial (identical results to ~1e-6 across kernels). A string literal
    // (`"simd"`/`"amx"`/`"auto"`/`"punned"`); unset = `Auto`.
    if (!object.isLiteral()) {
      throw VectorSearchException{
          "The parameter `<bf16Kernel>` expects a string "
          "(`\"simd\"`, `\"amx\"`, `\"auto\"`, or `\"punned\"`)."};
    }
    std::string v{asStringViewUnsafe(object.getLiteral().getContent())};
    if (v == "auto") {
      bf16Kernel_ = qlever::vector::Bf16Kernel::Auto;
    } else if (v == "amx") {
      bf16Kernel_ = qlever::vector::Bf16Kernel::Amx;
    } else if (v == "simd") {
      bf16Kernel_ = qlever::vector::Bf16Kernel::Simd;
    } else if (v == "punned") {
      bf16Kernel_ = qlever::vector::Bf16Kernel::Punned;
    } else {
      throw VectorSearchException{absl::StrCat(
          "Unknown `<bf16Kernel>` value `", v,
          "`; expected `\"simd\"`, `\"amx\"`, `\"auto\"`, or `\"punned\"`. "
          "This is a performance A/B dial (identical results across kernels); "
          "an explicit kernel the CPU cannot run is silently downgraded.")};
    }
  } else {
    throw VectorSearchException{absl::StrCat(
        "Unsupported parameter `<", pred,
        ">` in vector search; supported: `<index>`, `<query>`, "
        "`<queryVector>`, `<queryText>`, `<imageUrl>`, `<imageBase64>`, "
        "`<candidates>` (alias `<left>`), `<result>`/`<right>`, "
        "`<bindScore>`, `<bindCoarseScore>`, `<bindCsls>`, `<k>` (alias "
        "`<numNearestNeighbors>`), `<rerankK>`, `<maxDistance>`, "
        "`<cslsThreshold>`, `<cslsNeighbors>`, `<autoCut>`, `<cutSignal>`, "
        "`<softmaxTemperature>`, `<softmaxN>`, `<algorithm>`, "
        "`<fullPrecision>`, `<bf16Kernel>`, `<i8Kernel>`.")};
  }
}

// ____________________________________________________________________________
void VectorSearchQuery::validate() const {
  // Throws on any inconsistency.
  std::ignore = toVectorSearchConfiguration();
}

// ____________________________________________________________________________
qlever::vector::VectorSearchConfiguration
VectorSearchQuery::toVectorSearchConfiguration() const {
  auto throwIf = [](bool cond, const char* msg) {
    if (cond) throw VectorSearchException{msg};
  };
  throwIf(!indexName_.has_value(),
          "Vector search requires the `<index>` parameter.");

  int numQueryPoints = static_cast<int>(queryVector_.has_value()) +
                       static_cast<int>(queryEntityIri_.has_value()) +
                       static_cast<int>(queryText_.has_value()) +
                       static_cast<int>(queryImage_.has_value());
  throwIf(numQueryPoints > 1,
          "Vector search requires exactly one of `<queryVector>`, `<query>` (a "
          "constant entity IRI), `<queryText>`, or "
          "`<imageUrl>`/`<imageBase64>` as the query point.");
  throwIf(numQueryPoints == 0 && !leftVar_.has_value(),
          "Vector search requires exactly one of `<queryVector>`, `<query>` (a "
          "constant entity IRI), `<queryText>`, `<imageUrl>`/`<imageBase64>`, "
          "or `<candidates>` (alias `<left>`; a variable bound by the "
          "surrounding query).");
  throwIf(queryVector_.has_value() && queryVector_->empty(),
          "The `<queryVector>` parameter must contain at least one number.");

  // `<result>` is REQUIRED in every form.
  throwIf(!resultVar_.has_value(),
          "Vector search requires the `<result>` (alias `<right>`) parameter "
          "(the variable bound to each result entity).");

  // The three surface forms:
  //  * FORM W (WHOLE-INDEX): a query point, `<candidates>` omitted -- or
  //    present but UNBOUND by the surrounding query, which then must name
  //    the SAME variable as `<result>` (`?in == ?out`): the top-k nearest
  //    to the query point over the WHOLE index, bound to `<result>`.
  //  * FORM P (PRE-FILTER): a query point AND `<candidates> ?in` BOUND by
  //    the surrounding query: every bound candidate is scored by the
  //    distance of its STORED vector to the query point; the search is
  //    RESTRICTED to the bound set (it never pulls in non-candidates).
  //    `?in == ?out` ANNOTATES the candidates in place -- all of them, or
  //    only the top-`<k>` of the bound set if `<k>` is given; a distinct
  //    `?out` binds the top-k of the bound set (fresh binding).
  //  * FORM E (ENTITY-TO-ENTITY): `<candidates> ?in` bound, NO query point:
  //    for each bound candidate, the k nearest of its OWN stored vector,
  //    bound to a DISTINCT `<result>`.
  // Bound-ness of `<candidates>` is a planner/runtime property, so the
  // parse-time split is only "query point or not" (FORM E vs FORM W/P); the
  // `VectorSearchJoin` operation resolves W-vs-P when the planner does or
  // does not complete it with a subtree binding `?in`.
  const bool entityToEntityForm = leftVar_.has_value() && numQueryPoints == 0;
  const bool annotateForm =
      leftVar_.has_value() && numQueryPoints == 1 && leftVar_ == resultVar_;

  if (entityToEntityForm) {
    // A candidate's neighbours are OTHER entities; `?in == ?out` (annotating
    // in place) is only meaningful with a fixed query point (FORM P).
    throwIf(leftVar_ == resultVar_,
            "The `<candidates>`/`<left>` and `<result>` variables of a vector "
            "search without a query point (the entity-to-entity form) must be "
            "different. (`?in == ?out` requires a query point.)");
    // The coarse->rerank pass needs a fixed query point; FORM E binds one
    // exact (fine) score per row.
    throwIf(coarseScoreVar_.has_value(),
            "The `<bindCoarseScore>` parameter requires a query point; it is "
            "not supported in the entity-to-entity `<candidates>` form "
            "(no query point).");
  }
  // The CSLS-machinery cuts -- the fixed `<cslsThreshold>` and the dynamic
  // `<autoCut>` (mutually exclusive): each needs a fixed query point (FORM W
  // / FORM P; they are meaningless per-candidate in FORM E), and each scores
  // EVERY candidate (a full scan -- of the fine layer on a single-layer
  // index, of the coarse scan layer with a bounded fine rerank on a
  // two-layer one), so the top-k-style coarse-pass parameters and the HNSW
  // override contradict them.
  using CutSignal = qlever::vector::VectorSearchConfiguration::CutSignal;
  const bool hasCslsCut = cslsThreshold_.has_value() || autoCut_.has_value();
  const CutSignal signal = cutSignal_.value_or(CutSignal::Cosine);
  const bool softmaxSignal =
      autoCut_.has_value() && signal == CutSignal::Softmax;
  // The CSLS value (`vec:bindCsls`) is defined only by the fixed threshold or
  // the `csls` signal.
  const bool cutNeedsCslsValue =
      cslsThreshold_.has_value() ||
      (autoCut_.has_value() && signal == CutSignal::Csls);
  throwIf(cslsThreshold_.has_value() && autoCut_.has_value(),
          "The `<cslsThreshold>` (fixed CSLS cut) and `<autoCut>` (dynamic "
          "cut) parameters are mutually exclusive; use at most one of them.");
  throwIf(hasCslsCut && numQueryPoints == 0,
          "A `<cslsThreshold>`/`<autoCut>` cut requires a query point "
          "(`<queryVector>`, `<query>`, `<queryText>`, or "
          "`<imageUrl>`/`<imageBase64>`); it is not supported in the "
          "entity-to-entity `<candidates>` form (no query point).");
  throwIf(
      hasCslsCut &&
          algo_ == qlever::vector::VectorSearchConfiguration::Algorithm::Hnsw,
      "The `<cslsThreshold>`/`<autoCut>` cuts score EVERY candidate (a full "
      "scan of the fine or coarse layer), so they cannot be combined with "
      "`<algorithm>` `vectorSearch:hnsw`.");
  throwIf(hasCslsCut && coarseScoreVar_.has_value(),
          "The `<cslsThreshold>`/`<autoCut>` cuts do not expose coarse "
          "distances: on a two-layer index the coarse scan only preselects "
          "the fine rerank set (every returned score is a fine-layer "
          "cosine), so `<bindCoarseScore>` cannot be combined with them.");
  throwIf(hasCslsCut && rerankK_.has_value(),
          "The `<cslsThreshold>`/`<autoCut>` cuts size their own fine rerank "
          "(the per-index `cslsRerankFloor` serving setting, widened "
          "automatically as needed), so `<rerankK>` cannot be combined with "
          "them.");
  // `<cutSignal>` and the per-signal knobs are only valid with `<autoCut>`.
  throwIf(cutSignal_.has_value() && !autoCut_.has_value(),
          "The `<cutSignal>` parameter requires `<autoCut>` (it selects the "
          "dynamic cut's method; the fixed `<cslsThreshold>` has none).");
  throwIf(softmaxTemperature_.has_value() && !softmaxSignal,
          "The `<softmaxTemperature>` parameter requires `<cutSignal> "
          "\"softmax\"` (or the back-compat `<autoCut> \"softmax\"`).");
  throwIf(softmaxN_.has_value() && !softmaxSignal,
          "The `<softmaxN>` parameter requires `<cutSignal> \"softmax\"` (or "
          "the back-compat `<autoCut> \"softmax\"`).");
  throwIf(
      cslsVar_.has_value() && !cutNeedsCslsValue,
      "The `<bindCsls>` parameter requires `<cslsThreshold>` or `<cutSignal> "
      "\"csls\"` (the CSLS value is only computed for the CSLS cuts; the "
      "cosine/softmax cuts define none).");
  throwIf(cslsNeighbors_.has_value() && !hasCslsCut,
          "The `<cslsNeighbors>` parameter requires `<cslsThreshold>` or "
          "`<autoCut>`.");
  throwIf(cslsVar_.has_value() && cslsVar_ == resultVar_,
          "The `<bindCsls>` and `<result>` variables of a vector search must "
          "be different.");
  throwIf(cslsVar_.has_value() && leftVar_.has_value() && cslsVar_ == leftVar_,
          "The `<bindCsls>` and `<candidates>`/`<left>` variables of a "
          "vector search must be different.");
  throwIf(
      cslsVar_.has_value() && scoreVar_.has_value() && cslsVar_ == scoreVar_,
      "The `<bindCsls>` and `<bindScore>` variables of a vector search "
      "must be different.");
  throwIf(scoreVar_.has_value() && scoreVar_ == resultVar_,
          "The `<bindScore>` and `<result>` variables of a vector search must "
          "be different.");
  throwIf(
      scoreVar_.has_value() && leftVar_.has_value() && scoreVar_ == leftVar_,
      "The `<bindScore>` and `<candidates>`/`<left>` variables of a vector "
      "search must be different.");
  throwIf(coarseScoreVar_.has_value() && coarseScoreVar_ == resultVar_,
          "The `<bindCoarseScore>` and `<result>` variables of a vector "
          "search must be different.");
  throwIf(coarseScoreVar_.has_value() && leftVar_.has_value() &&
              coarseScoreVar_ == leftVar_,
          "The `<bindCoarseScore>` and `<candidates>`/`<left>` variables of a "
          "vector search must be different.");
  throwIf(coarseScoreVar_.has_value() && scoreVar_.has_value() &&
              coarseScoreVar_ == scoreVar_,
          "The `<bindCoarseScore>` and `<bindScore>` variables of a vector "
          "search must be different.");

  qlever::vector::VectorSearchConfiguration config;
  config.indexName_ = indexName_.value();
  config.queryVector_ = queryVector_;
  config.queryEntityIri_ = queryEntityIri_;
  config.queryText_ = queryText_;
  config.queryImage_ = queryImage_;
  config.leftVariable_ = leftVar_;
  config.resultVariable_ = resultVar_.value();
  config.scoreVariable_ = scoreVar_;
  config.coarseScoreVariable_ = coarseScoreVar_;
  config.k_ = k_.value_or(10);
  // FORM P annotate without an explicit `<k>` scores ALL bound candidates.
  config.keepAllCandidates_ = annotateForm && !k_.has_value();
  config.rerankK_ = rerankK_;
  config.maxDistance_ = maxDistance_;
  config.cslsThreshold_ = cslsThreshold_;
  config.cslsVariable_ = cslsVar_;
  config.cslsNeighbors_ = cslsNeighbors_;
  config.autoCut_ = autoCut_;
  config.cutSignal_ = cutSignal_.value_or(
      qlever::vector::VectorSearchConfiguration::CutSignal::Cosine);
  config.softmaxTemperature_ = softmaxTemperature_;
  config.softmaxN_ = softmaxN_;
  // With a CSLS-machinery cut (fixed or dynamic), an EXPLICIT `<k>` caps the
  // survivors; without one ALL survivors are returned (variable cardinality
  // -- the point of the cuts).
  config.cslsKCap_ = hasCslsCut && k_.has_value() ? k_ : std::nullopt;
  config.algorithm_ = algo_;
  config.fullPrecision_ = fullPrecision_;
  config.bf16Kernel_ = bf16Kernel_;
  config.i8Kernel_ = i8Kernel_;
  return config;
}

}  // namespace parsedQuery
