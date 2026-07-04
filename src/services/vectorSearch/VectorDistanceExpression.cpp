// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorDistanceExpression.h"

#include <absl/strings/charconv.h>
#include <absl/strings/str_cat.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <cmath>
#include <limits>
#include <variant>

#include "engine/sparqlExpressions/LiteralExpression.h"
#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "index/Index.h"
#include "index/IndexImpl.h"
#include "parser/NormalizedString.h"
#include "parser/TripleComponent.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "services/vectorSearch/VectorQueryPoint.h"
#include "util/Exception.h"

namespace sparqlExpression {

namespace {
// How often the per-row loop polls the cancellation handle -- frequent enough
// for prompt cancellation, rare enough to stay out of the hot path.
constexpr size_t CHECK_INTERRUPT_PERIOD = 65536;

// Parse a comma-separated list of finite floats, e.g. "0.1,-0.2,0.3" (the
// literal form of an explicit query vector). Throws on anything that is not a
// finite number.
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
      AD_THROW(absl::StrCat(
          "vec:distance: the query vector contains a value that is not a "
          "finite number: \"",
          token, "\"."));
    }
    out.push_back(value);
  }
  if (out.empty()) {
    AD_THROW(
        "vec:distance: the query vector literal must contain at least one "
        "number.");
  }
  return out;
}

// Extract the content of a constant string-literal argument, or `nullopt` if
// `expr` is not a string literal.
std::optional<std::string> getStringLiteral(const SparqlExpression* expr) {
  if (const auto* lit = dynamic_cast<const StringLiteralExpression*>(expr)) {
    return std::string{asStringViewUnsafe(lit->value().getContent())};
  }
  return std::nullopt;
}
}  // namespace

// _____________________________________________________________________________
VectorDistanceExpression::VectorDistanceExpression(
    std::string indexName, Ptr entity,
    std::optional<std::vector<float>> queryVector,
    std::optional<ad_utility::triple_component::Iri> queryEntityIri,
    std::optional<std::string> queryText, std::optional<ImageQuery> queryImage)
    : indexName_{std::move(indexName)},
      entity_{std::move(entity)},
      queryVector_{std::move(queryVector)},
      queryEntityIri_{std::move(queryEntityIri)},
      queryText_{std::move(queryText)},
      queryImage_{std::move(queryImage)} {
  // Exactly one query-point form must be set.
  AD_CONTRACT_CHECK(static_cast<int>(queryVector_.has_value()) +
                        static_cast<int>(queryEntityIri_.has_value()) +
                        static_cast<int>(queryText_.has_value()) +
                        static_cast<int>(queryImage_.has_value()) ==
                    1);
}

// _____________________________________________________________________________
ExpressionResult VectorDistanceExpression::evaluate(
    EvaluationContext* context) const {
  const Index& index = context->_qec.getIndex();
  auto vectorIndex = qlever::vector::getVectorIndex(index, indexName_);
  if (!vectorIndex) {
    AD_THROW(absl::StrCat("vec:distance: no vector index named \"", indexName_,
                          "\" is loaded."));
  }

  // Encode the (constant) query point ONCE into a reusable distance functor. A
  // text/image query point is embedded (once, cached in `embeddedQuery_`) via
  // the index's own endpoint -- reusing `resolveQueryPoint`, the same path the
  // search SERVICE uses, so the query is embedded with the model that produced
  // the stored vectors.
  std::optional<qlever::vector::VectorIndex::DistanceComputer> computer;
  if (queryVector_.has_value()) {
    // Throws on a dimension mismatch with the index.
    computer = vectorIndex->makeDistanceComputer(queryVector_.value());
  } else if (queryEntityIri_.has_value()) {
    TripleComponent tc{queryEntityIri_.value()};
    std::optional<Id> queryId = tc.toValueId(index);
    if (!queryId.has_value()) {
      AD_THROW(absl::StrCat("vec:distance: the query entity ",
                            queryEntityIri_->toStringRepresentation(),
                            " is not in the knowledge graph."));
    }
    computer = vectorIndex->makeDistanceComputerByEntity(queryId.value());
    if (!computer.has_value()) {
      AD_THROW(absl::StrCat("vec:distance: the query entity ",
                            queryEntityIri_->toStringRepresentation(),
                            " has no vector in index \"", indexName_, "\"."));
    }
  } else {
    // Text or image query point -> embed ONCE, then cache the vector.
    if (!embeddedQuery_.has_value()) {
      qlever::vector::VectorSearchConfiguration config;
      config.indexName_ = indexName_;
      config.queryText_ = queryText_;
      config.queryImage_ = queryImage_;
      qlever::vector::QueryPoint point = qlever::vector::resolveQueryPoint(
          config, *vectorIndex, index.getImpl(), context->cancellationHandle_);
      // A text/image query point always resolves to an (embedded) vector, or
      // `resolveQueryPoint` throws (no endpoint / dimension mismatch).
      embeddedQuery_ = std::get<std::vector<float>>(std::move(point));
    }
    computer = vectorIndex->makeDistanceComputer(embeddedQuery_.value());
  }
  const auto& computeDistance = computer.value();

  const size_t resultSize = context->size();
  VectorWithMemoryLimit<Id> out{context->_allocator};
  out.reserve(resultSize);

  // Map a single evaluated operand element to the entity `Id`. Only a plain
  // vocabulary/blank-node id can have a stored vector; a literal or local-vocab
  // entity never does, so it maps to `Undefined` (-> `UNDEF` distance).
  auto toEntityId = [](const auto& element) -> Id {
    using E = std::decay_t<decltype(element)>;
    if constexpr (std::is_same_v<E, ValueId>) {
      return element;
    } else if constexpr (std::is_same_v<E, IdOrLocalVocabEntry>) {
      const ValueId* id = std::get_if<ValueId>(&element);
      return id != nullptr ? *id : Id::makeUndefined();
    } else {
      return Id::makeUndefined();
    }
  };

  size_t sinceCheck = 0;
  auto appendDistances = [&](const auto& input) {
    for (const auto& element :
         detail::makeGenerator(input, resultSize, context)) {
      Id entity = toEntityId(element);
      Id result = Id::makeUndefined();
      if (!entity.isUndefined()) {
        float distance = computeDistance(entity);
        if (!std::isnan(distance)) {
          result = Id::makeFromDouble(static_cast<double>(distance));
        }
      }
      out.push_back(result);
      if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
        sinceCheck = 0;
        context->cancellationHandle_->throwIfCancelled();
      }
    }
  };

  ExpressionResult entityResult = entity_->evaluate(context);
  std::visit([&](auto&& input) { appendDistances(input); },
             std::move(entityResult));
  return out;
}

// _____________________________________________________________________________
std::string VectorDistanceExpression::getCacheKey(
    const VariableToColumnMap& varColMap) const {
  std::string query;
  if (queryVector_.has_value()) {
    query = absl::StrCat("v:", absl::StrJoin(queryVector_.value(), ","));
  } else if (queryEntityIri_.has_value()) {
    query = absl::StrCat("e:", queryEntityIri_->toStringRepresentation());
  } else if (queryText_.has_value()) {
    query = absl::StrCat("t:", queryText_->size(), ":", queryText_.value());
  } else {
    query = absl::StrCat("i:", static_cast<int>(queryImage_->kind_), ":",
                         queryImage_->value_.size(), ":", queryImage_->value_);
  }
  return absl::StrCat("VEC_DISTANCE(", indexName_, "|",
                      entity_->getCacheKey(varColMap), "|", query, ")");
}

// _____________________________________________________________________________
ql::span<SparqlExpression::Ptr> VectorDistanceExpression::childrenImpl() {
  return {&entity_, 1};
}

namespace {
// Shared front of the three factories: check arity == 3 and extract the
// index-name string literal (argument 0).
std::string parseIndexName(const std::vector<SparqlExpression::Ptr>& args,
                           std::string_view fn) {
  if (args.size() != 3) {
    AD_THROW(absl::StrCat(fn,
                          " takes three arguments: an index-name string "
                          "literal, the entity (usually a variable), and the "
                          "query point; got ",
                          args.size(), "."));
  }
  std::optional<std::string> indexName = getStringLiteral(args[0].get());
  if (!indexName.has_value()) {
    AD_THROW(absl::StrCat(
        fn,
        ": the first argument must be a string literal naming the vector "
        "index, e.g. ",
        fn, "(\"clip\", ?e, ...)."));
  }
  return std::move(indexName).value();
}
}  // namespace

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorDistanceExpression(
    std::vector<SparqlExpression::Ptr> args) {
  std::string indexName = parseIndexName(args, "vec:distance");
  // The third argument is the constant query point: either a float-list string
  // literal, or an entity IRI whose stored vector is the query.
  std::optional<std::vector<float>> queryVector;
  std::optional<ad_utility::triple_component::Iri> queryEntityIri;
  if (auto floats = getStringLiteral(args[2].get()); floats.has_value()) {
    queryVector = parseFloatList(floats.value());
  } else if (const auto* iri =
                 dynamic_cast<const IriExpression*>(args[2].get())) {
    queryEntityIri = iri->value();
  } else {
    AD_THROW(
        "vec:distance: the third argument (the query point) must be a "
        "comma-separated float-list string literal (e.g. \"0.1,0.2,0.3\") or a "
        "constant entity IRI.");
  }
  return std::make_unique<VectorDistanceExpression>(
      std::move(indexName), std::move(args[1]), std::move(queryVector),
      std::move(queryEntityIri), std::nullopt, std::nullopt);
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorDistanceTextExpression(
    std::vector<SparqlExpression::Ptr> args) {
  std::string indexName = parseIndexName(args, "vec:distanceText");
  std::optional<std::string> text = getStringLiteral(args[2].get());
  if (!text.has_value()) {
    AD_THROW(
        "vec:distanceText: the third argument must be a string literal of the "
        "query text to embed, e.g. vec:distanceText(\"clip\", ?e, \"a green "
        "statue\").");
  }
  return std::make_unique<VectorDistanceExpression>(
      std::move(indexName), std::move(args[1]), std::nullopt, std::nullopt,
      std::move(text), std::nullopt);
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorDistanceImageExpression(
    std::vector<SparqlExpression::Ptr> args) {
  using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
  std::string indexName = parseIndexName(args, "vec:distanceImage");
  // The image query point is either a URL (an IRI, fetched by the endpoint) or
  // a base64 payload (a string literal).
  std::optional<VectorDistanceExpression::ImageQuery> image;
  if (const auto* iri = dynamic_cast<const IriExpression*>(args[2].get())) {
    image = {ImageKind::Url, iri->value().toStringRepresentation()};
  } else if (auto b64 = getStringLiteral(args[2].get()); b64.has_value()) {
    image = {ImageKind::Base64, std::move(b64).value()};
  } else {
    AD_THROW(
        "vec:distanceImage: the third argument must be an image URL (a "
        "constant IRI) or a base64 string literal.");
  }
  return std::make_unique<VectorDistanceExpression>(
      std::move(indexName), std::move(args[1]), std::nullopt, std::nullopt,
      std::nullopt, std::move(image));
}

}  // namespace sparqlExpression
