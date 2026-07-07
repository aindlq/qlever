// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorEmbedExpression.h"

#include <absl/strings/str_cat.h>

#include <optional>
#include <utility>
#include <variant>

#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "index/ExportIds.h"
#include "index/Index.h"
#include "index/IndexImpl.h"
#include "parser/NormalizedString.h"
#include "rdfTypes/Iri.h"
#include "services/vectorSearch/EmbeddingClient.h"
#include "services/vectorSearch/VectorDistanceExpression.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/Exception.h"
#include "util/Log.h"
#include "util/Timer.h"

namespace sparqlExpression {

namespace {
// How often the per-row loop polls the cancellation handle. (Each MISSING
// cache entry additionally pays a whole embedding request, which polls the
// handle itself.)
constexpr size_t CHECK_INTERRUPT_PERIOD = 1024;

// What to embed, resolved from one evaluated input element: a literal is
// embedded as TEXT, an IRI as an IMAGE URL (or `data:` URI).
struct EmbedInput {
  bool isImage_;
  std::string value_;
};

// Resolve one evaluated input element (a `ValueId` or an
// `IdOrLocalVocabEntry`) to an `EmbedInput`, or `nullopt` for anything that is
// neither a literal nor an IRI (-> `UNDEF`).
template <typename E>
std::optional<EmbedInput> getEmbedInput(const E& element,
                                        const EvaluationContext* context) {
  using T = std::decay_t<E>;
  static_assert(std::is_same_v<T, ValueId> ||
                std::is_same_v<T, IdOrLocalVocabEntry>);
  std::optional<ad_utility::triple_component::LiteralOrIri> term;
  auto fromId = [context](ValueId id) {
    return ql::exportIds::idToLiteralOrIri(context->_qec.getIndex().getImpl(),
                                           id, context->_localVocab);
  };
  if constexpr (std::is_same_v<T, ValueId>) {
    term = fromId(element);
  } else {
    if (const ValueId* id = std::get_if<ValueId>(&element)) {
      term = fromId(*id);
    } else {
      term = static_cast<const ad_utility::triple_component::LiteralOrIri&>(
          std::get<LocalVocabEntry>(element));
    }
  }
  if (!term.has_value()) {
    return std::nullopt;
  }
  std::string value{asStringViewUnsafe(term->getContent())};
  return EmbedInput{term->isIri(), std::move(value)};
}
}  // namespace

// _____________________________________________________________________________
VectorEmbedExpression::VectorEmbedExpression(std::string indexName, Ptr input)
    : indexName_{std::move(indexName)}, input_{std::move(input)} {}

// _____________________________________________________________________________
ExpressionResult VectorEmbedExpression::evaluate(
    EvaluationContext* context) const {
  const Index& index = context->_qec.getIndex();
  auto vectorIndex = qlever::vector::getVectorIndex(index, indexName_);
  if (!vectorIndex) {
    AD_THROW(absl::StrCat("vec:embed: no vector index named \"", indexName_,
                          "\" is loaded."));
  }
  const auto& config = vectorIndex->metadata().config_;
  if (config.embeddingUrl_.empty()) {
    AD_THROW(absl::StrCat(
        "vec:embed: vector index \"", indexName_,
        "\" has no embeddingUrl configured, so there is no embedding endpoint "
        "to embed the query with."));
  }
  // The result literals are TYPED with the index's embedding space (model +
  // storage precision), so `vec:distance` can validate comparability -- also
  // against a DIFFERENT index (see `VEC_QUERY_DATATYPE_PREFIX`).
  const auto datatype =
      ad_utility::triple_component::Iri::fromIrirefWithoutBrackets(
          qlever::vector::vectorQueryDatatypeIri(
              config.embeddingModel_,
              qlever::vector::toString(config.scalar_)));

  // The accumulated wall time of the REAL embedding requests of this
  // `evaluate` block (cache hits cost nothing and are not counted), logged
  // once below -- never per row.
  ad_utility::Timer requestTimer{ad_utility::Timer::Stopped};
  size_t numRequests = 0;

  // Embed one input, MEMOIZED by (kind, value): a constant input embeds once,
  // a per-row variable embeds once per distinct value.
  auto embedMemoized = [&](const EmbedInput& input) -> const std::string& {
    std::string key = absl::StrCat(input.isImage_ ? "i:" : "t:", input.value_);
    if (auto it = cache_.find(key); it != cache_.end()) {
      return it->second;
    }
    requestTimer.cont();
    std::vector<float> embedding =
        input.isImage_ ? qlever::vector::embedImageOpenAI(
                             config.embeddingUrl_, config.embeddingModel_,
                             input.value_, context->cancellationHandle_)
                       : qlever::vector::embedTextOpenAI(
                             config.embeddingUrl_, config.embeddingModel_,
                             input.value_, context->cancellationHandle_);
    requestTimer.stop();
    ++numRequests;
    return cache_
        .emplace(std::move(key), qlever::vector::toFloatListString(embedding))
        .first->second;
  };

  // Turn one evaluated element into the result entry.
  auto embedElement = [&](const auto& element) -> IdOrLocalVocabEntry {
    std::optional<EmbedInput> input = getEmbedInput(element, context);
    if (!input.has_value()) {
      return Id::makeUndefined();
    }
    return LocalVocabEntry{
        ad_utility::triple_component::Literal::literalWithoutQuotes(
            embedMemoized(input.value()), datatype),
        context->getLocalVocabContext()};
  };

  ExpressionResult inputResult = input_->evaluate(context);
  ExpressionResult result = std::visit(
      [&](auto&& input) -> ExpressionResult {
        using T = std::decay_t<decltype(input)>;
        if constexpr (isConstantResult<T>) {
          // A constant input -> one constant literal (embedded once).
          return embedElement(input);
        } else {
          const size_t resultSize = context->size();
          VectorWithMemoryLimit<IdOrLocalVocabEntry> out{context->_allocator};
          out.reserve(resultSize);
          size_t sinceCheck = 0;
          for (const auto& element :
               detail::makeGenerator(AD_FWD(input), resultSize, context)) {
            out.push_back(embedElement(element));
            if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
              sinceCheck = 0;
              context->cancellationHandle_->throwIfCancelled();
            }
          }
          return out;
        }
      },
      std::move(inputResult));
  // Only log when at least one REAL endpoint round-trip happened; a block
  // served purely from the memoization cache stays silent.
  if (numRequests > 0) {
    AD_LOG_INFO << "vec:embed[" << indexName_ << "]: embedded " << numRequests
                << " input(s) in " << requestTimer.msecs().count() << " ms via "
                << config.embeddingUrl_ << std::endl;
  }
  return result;
}

// _____________________________________________________________________________
std::string VectorEmbedExpression::getCacheKey(
    const VariableToColumnMap& varColMap) const {
  return absl::StrCat("VEC_EMBED(", indexName_, "|",
                      input_->getCacheKey(varColMap), ")");
}

// _____________________________________________________________________________
ql::span<SparqlExpression::Ptr> VectorEmbedExpression::childrenImpl() {
  return {&input_, 1};
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorEmbedExpression(
    std::vector<SparqlExpression::Ptr> args) {
  if (args.size() != 2) {
    AD_THROW(absl::StrCat(
        "vec:embed takes two arguments: the index IRI (",
        qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
        "NAME>) and the input to embed (a text literal or an image IRI); "
        "got ",
        args.size(), "."));
  }
  std::string indexName =
      parseVectorIndexIriArgument(args[0].get(), "vec:embed");
  return std::make_unique<VectorEmbedExpression>(std::move(indexName),
                                                 std::move(args[1]));
}

}  // namespace sparqlExpression
