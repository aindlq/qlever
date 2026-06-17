// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.
// Author: Artem <artem@rem.sh>

#include "services/vectorSearch/VectorSearchQuery.h"

#include <charconv>

#include "parser/SparqlTriple.h"

namespace parsedQuery {

namespace {
// Parse a comma-separated list of floats, e.g. "0.1,-0.2,0.3".
std::vector<float> parseFloatList(std::string_view s) {
  std::vector<float> out;
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string_view tok =
        s.substr(start, comma == std::string_view::npos ? std::string_view::npos
                                                        : comma - start);
    // Trim spaces.
    while (!tok.empty() && (tok.front() == ' ' || tok.front() == '\t'))
      tok.remove_prefix(1);
    while (!tok.empty() && (tok.back() == ' ' || tok.back() == '\t'))
      tok.remove_suffix(1);
    if (!tok.empty()) {
      // std::from_chars for float isn't available on all stdlibs we target, so
      // use std::stof via a temporary string.
      out.push_back(std::stof(std::string{tok}));
    }
    if (comma == std::string_view::npos) break;
    start = comma + 1;
  }
  return out;
}
}  // namespace

void VectorSearchQuery::addParameter(const SparqlTriple& triple) {
  auto simpleTriple = triple.getSimple();
  TripleComponent predicate = simpleTriple.p_;
  TripleComponent object = simpleTriple.o_;
  auto pred = extractParameterName(predicate, VECTOR_SEARCH_IRI);

  auto requireLiteral = [&object, &pred]() -> std::string {
    if (!object.isLiteral()) {
      throw VectorSearchException{absl::StrCat(
          "The parameter `<", pred, ">` expects a string literal.")};
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
  } else if (pred == "imageFile") {
    using ImageKind = qlever::vector::VectorSearchConfiguration::ImageKind;
    queryImage_ = {ImageKind::File, requireLiteral()};
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
        "`<queryVector>`, `<left>`, `<result>`/`<right>`, `<bindScore>`, `<k>`, "
        "`<algorithm>`.")};
  }
}

void VectorSearchQuery::validate() const {
  // Throws on any inconsistency.
  std::ignore = toVectorSearchConfiguration();
}

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
          "constant entity IRI), `<queryText>`, `<imageUrl>`/`<imageFile>`/"
          "`<imageBase64>`, or `<left>` (a variable bound by a nested pattern).");

  if (leftVar_.has_value()) {
    // Binary "for each ?x" form: the nested pattern binds the query entities.
    throwIf(!childGraphPattern_.has_value(),
            "The `<left>` form of vector search requires a nested `{ ... }` "
            "graph pattern that binds the query variable.");
  }
  // Otherwise (a query *point* via queryVector/query/queryText): a nested
  // pattern is OPTIONAL and, if present, restricts the search to the entities it
  // binds to `<result>` (exact search over that candidate set -- the "small
  // candidate set" optimisation). The result-variable binding is checked when
  // the operation is constructed.

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
  config.algorithm_ = algo_;
  return config;
}

}  // namespace parsedQuery
