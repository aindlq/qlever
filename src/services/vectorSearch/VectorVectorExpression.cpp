// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorVectorExpression.h"

#include <absl/strings/str_cat.h>

#include <optional>
#include <utility>
#include <variant>

#include "engine/sparqlExpressions/SparqlExpressionGenerators.h"
#include "rdfTypes/Iri.h"
#include "rdfTypes/Literal.h"
#include "services/vectorSearch/VectorDistanceExpression.h"
#include "services/vectorSearch/VectorIndex.h"
#include "services/vectorSearch/VectorIndexExtension.h"
#include "util/Exception.h"

namespace sparqlExpression {

namespace {
// How often the per-row loop polls the cancellation handle (each row pays a
// rowmap lookup plus a row decode + serialization, so poll reasonably often).
constexpr size_t CHECK_INTERRUPT_PERIOD = 8192;

// Extract the `ValueId` of one evaluated element (a `ValueId` or an
// `IdOrLocalVocabEntry`), or `nullptr` if the element holds a fresh
// LocalVocab entry (which can never be a stored entity of the index).
template <typename E>
const ValueId* getEntityId(const E& element) {
  using T = std::decay_t<E>;
  static_assert(std::is_same_v<T, ValueId> ||
                std::is_same_v<T, IdOrLocalVocabEntry>);
  if constexpr (std::is_same_v<T, ValueId>) {
    return &element;
  } else {
    return std::get_if<ValueId>(&element);
  }
}
}  // namespace

// _____________________________________________________________________________
VectorVectorExpression::VectorVectorExpression(std::string indexName,
                                               Ptr entity)
    : indexName_{std::move(indexName)}, entity_{std::move(entity)} {}

// _____________________________________________________________________________
ExpressionResult VectorVectorExpression::evaluate(
    EvaluationContext* context) const {
  auto vectorIndex =
      qlever::vector::getVectorIndex(context->_qec.getIndex(), indexName_);
  if (!vectorIndex) {
    AD_THROW(absl::StrCat("vec:vector: no vector index named \"", indexName_,
                          "\" is loaded."));
  }
  const qlever::vector::VectorIndex& vidx = *vectorIndex;
  // The result literals are TYPED with THIS index's embedding space (model +
  // storage precision), so `vec:distance` can validate comparability -- also
  // against a DIFFERENT index (see `VEC_QUERY_DATATYPE_PREFIX`).
  const auto& config = vidx.metadata().config_;
  const auto datatype =
      ad_utility::triple_component::Iri::fromIrirefWithoutBrackets(
          qlever::vector::vectorQueryDatatypeIri(
              config.embeddingModel_,
              qlever::vector::toString(config.scalar_)));

  // Turn one evaluated element into the result entry: the entity's stored
  // vector decoded to f32 as a typed float-list literal, or UNDEF for a
  // non-member (the same implicit membership filter as `vec:distance`).
  auto vectorElement = [&](const auto& element) -> IdOrLocalVocabEntry {
    const ValueId* id = getEntityId(element);
    if (id == nullptr) {
      return Id::makeUndefined();
    }
    std::optional<std::vector<float>> floats = vidx.getVector(*id);
    if (!floats.has_value()) {
      return Id::makeUndefined();
    }
    return LocalVocabEntry{
        ad_utility::triple_component::Literal::literalWithoutQuotes(
            qlever::vector::toFloatListString(floats.value()), datatype),
        context->getLocalVocabContext()};
  };

  ExpressionResult entityResult = entity_->evaluate(context);
  return std::visit(
      [&](auto&& input) -> ExpressionResult {
        using T = std::decay_t<decltype(input)>;
        if constexpr (isConstantResult<T>) {
          // A constant entity -> one constant literal (looked up once).
          return vectorElement(input);
        } else {
          const size_t resultSize = context->size();
          VectorWithMemoryLimit<IdOrLocalVocabEntry> out{context->_allocator};
          out.reserve(resultSize);
          size_t sinceCheck = 0;
          for (const auto& element :
               detail::makeGenerator(AD_FWD(input), resultSize, context)) {
            out.push_back(vectorElement(element));
            if (++sinceCheck == CHECK_INTERRUPT_PERIOD) {
              sinceCheck = 0;
              context->cancellationHandle_->throwIfCancelled();
            }
          }
          return out;
        }
      },
      std::move(entityResult));
}

// _____________________________________________________________________________
std::string VectorVectorExpression::getCacheKey(
    const VariableToColumnMap& varColMap) const {
  return absl::StrCat("VEC_VECTOR(", indexName_, "|",
                      entity_->getCacheKey(varColMap), ")");
}

// _____________________________________________________________________________
ql::span<SparqlExpression::Ptr> VectorVectorExpression::childrenImpl() {
  return {&entity_, 1};
}

// _____________________________________________________________________________
SparqlExpression::Ptr makeVectorVectorExpression(
    std::vector<SparqlExpression::Ptr> args) {
  if (args.size() != 2) {
    AD_THROW(
        absl::StrCat("vec:vector takes two arguments: the index IRI (",
                     qlever::vector::VECTOR_METADATA_SUBJECT_PREFIX,
                     "NAME>) and the entity whose stored vector to fetch; got ",
                     args.size(), "."));
  }
  std::string indexName =
      parseVectorIndexIriArgument(args[0].get(), "vec:vector");
  return std::make_unique<VectorVectorExpression>(std::move(indexName),
                                                  std::move(args[1]));
}

}  // namespace sparqlExpression
