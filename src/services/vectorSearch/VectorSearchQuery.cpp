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
    indexName_ = requireLiteral();
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
    std::string url = object.isIri() ? object.getIri().toStringRepresentation()
                                     : requireLiteral();
    queryImage_ = {ImageKind::Url, std::move(url)};
  } else if (pred == "imageBase64") {
    using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
    queryImage_ = {ImageKind::Base64, requireLiteral()};
  } else if (pred == "left") {
    setVariable("left", object, leftVar_);
  } else if (pred == "result" || pred == "right") {
    setVariable("result", object, resultVar_);
  } else if (pred == "bindScore") {
    setVariable("bindScore", object, scoreVar_);
  } else if (pred == "k" || pred == "numNearestNeighbors") {
    if (!object.isInt() || object.getInt() <= 0) {
      throw VectorSearchException{
          "The parameter `<k>` expects a positive integer."};
    }
    k_ = static_cast<size_t>(object.getInt());
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
  } else {
    throw VectorSearchException{absl::StrCat(
        "Unsupported parameter `<", pred,
        ">` in vector search; supported: `<index>`, `<query>`, "
        "`<queryVector>`, `<queryText>`, `<imageUrl>`, `<imageBase64>`, "
        "`<left>`, `<result>`/`<right>`, `<bindScore>`, `<k>` (alias "
        "`<numNearestNeighbors>`), `<maxDistance>`, `<algorithm>`.")};
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
  throwIf(!resultVar_.has_value(),
          "Vector search requires the `<result>` (alias `<right>`) parameter "
          "(the variable bound to each result entity).");

  int numQuerySpecs = static_cast<int>(queryVector_.has_value()) +
                      static_cast<int>(queryEntityIri_.has_value()) +
                      static_cast<int>(queryText_.has_value()) +
                      static_cast<int>(queryImage_.has_value()) +
                      static_cast<int>(leftVar_.has_value());
  throwIf(numQuerySpecs != 1,
          "Vector search requires exactly one of `<queryVector>`, `<query>` (a "
          "constant entity IRI), `<queryText>`, `<imageUrl>`/`<imageBase64>`, "
          "or `<left>` (a variable bound by the surrounding query).");
  throwIf(queryVector_.has_value() && queryVector_->empty(),
          "The `<queryVector>` parameter must contain at least one number.");

  if (leftVar_.has_value()) {
    // Binary "for each ?x" form: the query entities are bound by the
    // SURROUNDING query (the planner joins the vector search with the subtree
    // that binds the variable).
    throwIf(leftVar_ == resultVar_,
            "The `<left>` and `<result>` variables of a vector search must be "
            "different.");
  }
  throwIf(scoreVar_.has_value() && scoreVar_ == resultVar_,
          "The `<bindScore>` and `<result>` variables of a vector search must "
          "be different.");
  throwIf(
      scoreVar_.has_value() && leftVar_.has_value() && scoreVar_ == leftVar_,
      "The `<bindScore>` and `<left>` variables of a vector search must be "
      "different.");

  qlever::vector::VectorSearchConfiguration config;
  config.indexName_ = indexName_.value();
  config.queryVector_ = queryVector_;
  config.queryEntityIri_ = queryEntityIri_;
  config.queryText_ = queryText_;
  config.queryImage_ = queryImage_;
  config.leftVariable_ = leftVar_;
  config.resultVariable_ = resultVar_.value();
  config.scoreVariable_ = scoreVar_;
  config.k_ = k_.value_or(10);
  config.maxDistance_ = maxDistance_;
  config.algorithm_ = algo_;
  return config;
}

}  // namespace parsedQuery
