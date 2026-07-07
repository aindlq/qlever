// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorDistanceExpression.h"

#include <absl/strings/charconv.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>

#include <cmath>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "backports/StartsWithAndEndsWith.h"
#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "index/ExportIds.h"
#include "index/Index.h"
#include "index/IndexImpl.h"
#include "parser/NormalizedString.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/Exception.h"
#include "util/Log.h"
#include "util/Timer.h"

namespace sparqlExpression {

namespace {
using qlever::vector::VectorIndex;

// How often the per-row loop polls the cancellation handle -- frequent enough
// for prompt cancellation, rare enough to stay out of the hot path.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// Parse a comma-separated list of finite floats, e.g. "0.1,-0.2,0.3" (the
// string form of an explicit query vector, and the output format of
// `vec:embed`). Returns `nullopt` (instead of throwing) on anything that is
// not a non-empty list of finite numbers, because per-row values that are not
// float lists simply resolve to `UNDEF`.
std::optional<std::vector<float>> tryParseFloatList(std::string_view s) {
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
      return std::nullopt;
    }
    out.push_back(value);
  }
  if (out.empty()) {
    return std::nullopt;
  }
  return out;
}

// A resolved vector source (one side of the distance):
//  * `monostate`     -> no vector (the row evaluates to `UNDEF`);
//  * `Id`            -> an entity with a stored vector in the index;
//  * `vector<float>` -> an explicit (parsed) query vector.
using ResolvedSource = std::variant<std::monostate, Id, std::vector<float>>;

// Whether a source is being resolved ONCE (a constant sub-expression) or per
// row. A constant that is a malformed or dimension-mismatched float list
// throws a loud, user-facing error (it is certainly a query mistake); per-row
// values never throw and resolve to `monostate` (-> `UNDEF`) instead, so a
// mixed column cannot abort a scan.
enum class SourceMode { Constant, PerRow };

// Resolve one evaluated source element (a `ValueId` or an
// `IdOrLocalVocabEntry`) to a `ResolvedSource`: the entity path first (a
// cheap rowmap lookup), then the float-list path for literal content.
template <typename E>
ResolvedSource resolveSource(const E& element, const VectorIndex& vidx,
                             const std::string& indexName,
                             const EvaluationContext* context,
                             SourceMode mode) {
  using T = std::decay_t<E>;
  static_assert(std::is_same_v<T, ValueId> ||
                std::is_same_v<T, IdOrLocalVocabEntry>);

  // The entity path: a value id whose entity has a (live) stored vector.
  const ValueId* id = nullptr;
  if constexpr (std::is_same_v<T, ValueId>) {
    id = &element;
  } else {
    id = std::get_if<ValueId>(&element);
  }
  if (id != nullptr) {
    if (id->isUndefined()) {
      return std::monostate{};
    }
    if (vidx.hasVector(*id)) {
      return *id;
    }
  }

  // The float-list path: literal content that parses as floats (an inline
  // "0.1,0.2,..." or the output of `vec:embed`). An IRI/entity WITHOUT a
  // stored vector also ends up here and resolves to `monostate`, because an
  // IRI never parses as a float list.
  std::optional<ad_utility::triple_component::LiteralOrIri> term;
  if (id != nullptr) {
    term = ql::exportIds::idToLiteralOrIri(context->_qec.getIndex().getImpl(),
                                           *id, context->_localVocab);
  } else if constexpr (std::is_same_v<T, IdOrLocalVocabEntry>) {
    term = static_cast<const ad_utility::triple_component::LiteralOrIri&>(
        std::get<LocalVocabEntry>(element));
  }
  if (!term.has_value() || !term->isLiteral()) {
    return std::monostate{};
  }
  const auto& literal = term->getLiteral();
  // A TYPED query-vector literal `"f0,f1,..."^^<.../vec/MODEL/PRECISION>` (the
  // output of `vec:embed`/`vec:vector` -- possibly of a DIFFERENT index)
  // carries its embedding space in the datatype. VALIDATE it against THIS
  // index before even parsing the floats: same-dimension vectors from
  // different models are numerically computable but semantically garbage, so
  // a space mismatch makes the row UNDEF (or throws for a constant source,
  // like the dimension error below). The model check is skipped when EITHER
  // side declares no model, so vector-only indices (no `embeddingModel`) stay
  // usable. A PLAIN untyped float-list string (an inline query vector) skips
  // all of this and is only dimension-checked below.
  if (literal.hasDatatype()) {
    std::string_view datatypeIri = asStringViewUnsafe(literal.getDatatype());
    if (ql::starts_with(datatypeIri,
                        qlever::vector::VEC_QUERY_DATATYPE_PREFIX)) {
      std::optional<std::pair<std::string, std::string>> space =
          qlever::vector::parseVectorQueryDatatypeIri(datatypeIri);
      const auto& config = vidx.metadata().config_;
      const std::string precision = qlever::vector::toString(config.scalar_);
      const bool modelComparable =
          space.has_value() &&
          (space->first.empty() || config.embeddingModel_.empty() ||
           space->first == config.embeddingModel_);
      if (!space.has_value() || space->second != precision ||
          !modelComparable) {
        if (mode == SourceMode::Constant) {
          AD_THROW(absl::StrCat(
              "vec:distance: the typed query vector's datatype <", datatypeIri,
              "> does not match vector index \"", indexName, "\" (model \"",
              config.embeddingModel_.empty() ? "<none>"
                                             : config.embeddingModel_,
              "\", precision \"", precision,
              "\"): vectors are only comparable within one embedding space "
              "(same model, same precision, same dimension)."));
        }
        return std::monostate{};
      }
    }
  }
  std::string_view content = asStringViewUnsafe(term->getContent());
  std::optional<std::vector<float>> floats = tryParseFloatList(content);
  if (!floats.has_value()) {
    if (mode == SourceMode::Constant) {
      AD_THROW(absl::StrCat(
          "vec:distance: the constant string \"", content,
          "\" is not a comma-separated list of finite numbers (a query "
          "vector), and no entity with a vector in index \"",
          indexName, "\" matches it."));
    }
    return std::monostate{};
  }
  if (floats->size() != vidx.dimensions()) {
    if (mode == SourceMode::Constant) {
      AD_THROW(absl::StrCat("vec:distance: the query vector has dimension ",
                            floats->size(), ", but vector index \"", indexName,
                            "\" has dimension ", vidx.dimensions(), "."));
    }
    return std::monostate{};
  }
  return std::move(floats).value();
}

// Convert a raw distance (`NaN` = no result) to the result `Id`.
Id toDistanceId(float distance) {
  return std::isnan(distance)
             ? Id::makeUndefined()
             : Id::makeFromDouble(static_cast<double>(distance));
}

// The one-shot distance between two resolved sources (used when neither side
// is a fixed query point that would pay for a reusable `DistanceComputer`).
Id pairDistance(const ResolvedSource& a, const ResolvedSource& b,
                const VectorIndex& vidx) {
  if (std::holds_alternative<std::monostate>(a) ||
      std::holds_alternative<std::monostate>(b)) {
    return Id::makeUndefined();
  }
  const Id* idA = std::get_if<Id>(&a);
  const Id* idB = std::get_if<Id>(&b);
  if (idA != nullptr && idB != nullptr) {
    return toDistanceId(vidx.distance(*idA, *idB));
  }
  if (idA != nullptr) {
    return toDistanceId(vidx.distance(*idA, std::get<std::vector<float>>(b)));
  }
  if (idB != nullptr) {
    return toDistanceId(vidx.distance(*idB, std::get<std::vector<float>>(a)));
  }
  return toDistanceId(vidx.distance(std::get<std::vector<float>>(a),
                                    std::get<std::vector<float>>(b)));
}
}  // namespace

// _____________________________________________________________________________
VectorDistanceExpression::VectorDistanceExpression(std::string indexName,
                                                   Ptr source1, Ptr source2)
    : indexName_{std::move(indexName)},
      sources_{std::move(source1), std::move(source2)} {}

// _____________________________________________________________________________
ExpressionResult VectorDistanceExpression::evaluate(
    EvaluationContext* context) const {
  const Index& index = context->_qec.getIndex();
  auto vectorIndex = qlever::vector::getVectorIndex(index, indexName_);
  if (!vectorIndex) {
    AD_THROW(absl::StrCat("vec:distance: no vector index named \"", indexName_,
                          "\" is loaded."));
  }
  const VectorIndex& vidx = *vectorIndex;
  const size_t resultSize = context->size();

  // How many rows of this `evaluate` block actually resolved to a distance
  // (UNDEF rows are not counted); drives the one INFO timing line below.
  size_t numDistances = 0;

  // Per-row source against a FIXED source: encode the fixed query point ONCE
  // into a reusable `DistanceComputer` (one rowmap lookup + one SIMD kernel
  // call per row).
  auto perRowAgainstConstant = [&](const ResolvedSource& constant,
                                   auto&& perRow) -> ExpressionResult {
    if (std::holds_alternative<std::monostate>(constant)) {
      // The fixed side resolves to no vector -> every row is UNDEF. (A
      // constant is a valid constant result, it broadcasts to all rows.)
      return Id::makeUndefined();
    }
    std::optional<VectorIndex::DistanceComputer> computer =
        std::holds_alternative<Id>(constant)
            ? vidx.makeDistanceComputerByEntity(std::get<Id>(constant))
            : std::optional{vidx.makeDistanceComputer(
                  std::get<std::vector<float>>(constant))};
    // `resolveSource` only returns an entity after checking `hasVector`.
    AD_CORRECTNESS_CHECK(computer.has_value());
    VectorWithMemoryLimit<Id> out{context->_allocator};
    out.reserve(resultSize);
    size_t sinceCheck = 0;
    for (const auto& element :
         detail::makeGenerator(AD_FWD(perRow), resultSize, context)) {
      ResolvedSource row =
          resolveSource(element, vidx, indexName_, context, SourceMode::PerRow);
      Id result = Id::makeUndefined();
      if (const Id* entity = std::get_if<Id>(&row)) {
        result = toDistanceId((*computer)(*entity));
      } else if (const auto* vec = std::get_if<std::vector<float>>(&row)) {
        result = toDistanceId((*computer)(*vec));
      }
      if (!result.isUndefined()) {
        ++numDistances;
      }
      out.push_back(result);
      if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        context->cancellationHandle_->throwIfCancelled();
      }
    }
    return out;
  };

  // Both sources per-row: zip the two generators and compute each pair
  // one-shot (entity<->entity uses the stored bytes directly, no copies).
  auto perRowPair = [&](auto&& perRow1, auto&& perRow2) -> ExpressionResult {
    VectorWithMemoryLimit<Id> out{context->_allocator};
    out.reserve(resultSize);
    auto gen1 = detail::makeGenerator(AD_FWD(perRow1), resultSize, context);
    auto gen2 = detail::makeGenerator(AD_FWD(perRow2), resultSize, context);
    auto it1 = gen1.begin();
    auto it2 = gen2.begin();
    size_t sinceCheck = 0;
    for (size_t i = 0; i < resultSize; ++i, ++it1, ++it2) {
      ResolvedSource a =
          resolveSource(*it1, vidx, indexName_, context, SourceMode::PerRow);
      ResolvedSource b =
          resolveSource(*it2, vidx, indexName_, context, SourceMode::PerRow);
      Id distance = pairDistance(a, b, vidx);
      if (!distance.isUndefined()) {
        ++numDistances;
      }
      out.push_back(distance);
      if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        context->cancellationHandle_->throwIfCancelled();
      }
    }
    return out;
  };

  ExpressionResult result1 = sources_[0]->evaluate(context);
  ExpressionResult result2 = sources_[1]->evaluate(context);

  // Time ONLY the distance computation below (row resolution + the
  // `DistanceComputer`/SIMD kernel calls) -- the child evaluation above (e.g.
  // a `vec:embed`) is timed and logged separately.
  ad_utility::Timer distanceTimer{ad_utility::Timer::Started};

  // Branch on constness: the expression framework yields a CONSTANT result
  // for a constant sub-expression, so a constant source (an inline float
  // list, a constant entity IRI, a `vec:embed` of constants, ...) is
  // parsed/looked up/encoded exactly once, never per row.
  ExpressionResult result = std::visit(
      [&](auto&& in1, auto&& in2) -> ExpressionResult {
        using T1 = std::decay_t<decltype(in1)>;
        using T2 = std::decay_t<decltype(in2)>;
        if constexpr (isConstantResult<T1> && isConstantResult<T2>) {
          // Both constant -> a single constant distance.
          ResolvedSource a = resolveSource(in1, vidx, indexName_, context,
                                           SourceMode::Constant);
          ResolvedSource b = resolveSource(in2, vidx, indexName_, context,
                                           SourceMode::Constant);
          Id distance = pairDistance(a, b, vidx);
          if (!distance.isUndefined()) {
            ++numDistances;
          }
          return distance;
        } else if constexpr (isConstantResult<T1>) {
          return perRowAgainstConstant(
              resolveSource(in1, vidx, indexName_, context,
                            SourceMode::Constant),
              AD_FWD(in2));
        } else if constexpr (isConstantResult<T2>) {
          // All supported metrics are symmetric, so the sides may swap.
          return perRowAgainstConstant(
              resolveSource(in2, vidx, indexName_, context,
                            SourceMode::Constant),
              AD_FWD(in1));
        } else {
          return perRowPair(AD_FWD(in1), AD_FWD(in2));
        }
      },
      std::move(result1), std::move(result2));
  // A block that computed no distance at all (everything UNDEF) stays silent.
  if (numDistances > 0) {
    AD_LOG_INFO << "vec:distance[" << indexName_ << "]: " << numDistances
                << " distances, dim " << vidx.dimensions() << " "
                << qlever::vector::toString(vidx.metadata().config_.scalar_)
                << ", in " << distanceTimer.msecs().count() << " ms"
                << std::endl;
  }
  return result;
}

// _____________________________________________________________________________
std::string VectorDistanceExpression::getCacheKey(
    const VariableToColumnMap& varColMap) const {
  return absl::StrCat("VEC_DISTANCE(", indexName_, "|",
                      sources_[0]->getCacheKey(varColMap), "|",
                      sources_[1]->getCacheKey(varColMap), ")");
}

// _____________________________________________________________________________
ql::span<SparqlExpression::Ptr> VectorDistanceExpression::childrenImpl() {
  return {sources_.data(), sources_.size()};
}

// _____________________________________________________________________________
std::string parseVectorIndexIriArgument(const SparqlExpression* arg,
                                        std::string_view functionName) {
  if (const auto* iri = dynamic_cast<const IriExpression*>(arg)) {
    std::optional<std::string> name = qlever::vector::indexNameFromMetadataIri(
        iri->value().toStringRepresentation());
    if (name.has_value()) {
      return std::move(name).value();
    }
  }
  AD_THROW(absl::StrCat(
      functionName,
      ": the first argument must be the index IRI -- the same resource that "
      "carries the index's metadata triples -- of the form ",
      qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX, "NAME>, e.g. ",
      functionName, "(", qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
      "clip>, ...)."));
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args) {
  if (args.size() != 3) {
    AD_THROW(absl::StrCat(
        "vec:distance takes three arguments: the index IRI (",
        qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
        "NAME>) and two vector sources, each of which is an entity with a "
        "stored vector or a comma-separated float-list string; got ",
        args.size(), "."));
  }
  std::string indexName =
      parseVectorIndexIriArgument(args[0].get(), "vec:distance");
  return std::make_unique<VectorDistanceExpression>(
      std::move(indexName), std::move(args[1]), std::move(args[2]));
}

}  // namespace sparqlExpression
