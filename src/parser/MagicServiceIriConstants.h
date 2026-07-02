// Copyright 2024, University of Freiburg
// Chair of Algorithms and Data Structures
// Author: Christoph Ullinger <ullingec@informatik.uni-freiburg.de>

#ifndef QLEVER_SRC_PARSER_MAGICSERVICEIRICONSTANTS_H
#define QLEVER_SRC_PARSER_MAGICSERVICEIRICONSTANTS_H

#include <ctre-unicode.hpp>
#include <string>
#include <string_view>

#include "util/StringUtils.h"

// Constants for the various magic services - they are invoked using these
// federated querying IRIs but actually never contact these and activate special
// query features locally

constexpr inline std::string_view PATH_SEARCH_IRI =
    "<https://qlever.cs.uni-freiburg.de/pathSearch/>";

constexpr inline std::string_view SPATIAL_SEARCH_IRI =
    "<https://qlever.cs.uni-freiburg.de/spatialSearch/>";

constexpr inline std::string_view TEXT_SEARCH_IRI =
    "<https://qlever.cs.uni-freiburg.de/textSearch/>";

constexpr inline std::string_view EXTERNAL_VALUES_IRI =
    "<https://qlever.cs.uni-freiburg.de/external-values/>";

// This prefix definition is used for backwards compatibility with the BMW use
// case (they already use this syntax in some of their applications which we
// do not want to break). New use cases of the `ExternalValues`
// features should use the `EXTERNAL_VALUES_IRI` above, because it is consistent
// with other magic service IRIs.
constexpr inline std::string_view EXTERNAL_VALUES_IRI_PREFIX =
    "<https://qlever.cs.uni-freiburg.de/external-values-";

namespace string_constants::detail {
constexpr inline std::string_view OPENING_BRACKET = "<";
constexpr inline std::string_view CLOSING_BRACKET = ">";
}  // namespace string_constants::detail

constexpr inline std::string_view MATERIALIZED_VIEW_IRI_WITHOUT_BRACKETS =
    "https://qlever.cs.uni-freiburg.de/materializedView/";
constexpr inline std::string_view
    MATERIALIZED_VIEW_IRI_WITHOUT_CLOSING_BRACKET =
        ad_utility::constexprStrCat<string_constants::detail::OPENING_BRACKET,
                                    MATERIALIZED_VIEW_IRI_WITHOUT_BRACKETS>();
constexpr inline std::string_view MATERIALIZED_VIEW_IRI =
    ad_utility::constexprStrCat<MATERIALIZED_VIEW_IRI_WITHOUT_CLOSING_BRACKET,
                                string_constants::detail::CLOSING_BRACKET>();

// The namespace under which QLever's magic `SERVICE`s live. A base
// magic-service IRI is `<MAGIC_SERVICE_IRI_NAMESPACE><serviceName>/>` (e.g.
// `vectorSearch/>`).
constexpr inline std::string_view MAGIC_SERVICE_IRI_NAMESPACE =
    "<https://qlever.cs.uni-freiburg.de/";

// Heuristic: does `iri` (bracketed representation) look like a magic-service
// base IRI -- i.e. the QLever namespace followed by a single path segment (a
// service name of `[A-Za-z0-9-]`) and a trailing `/`? Used to turn an
// unresolved magic-service IRI into a clear error rather than silently
// federating to a non-existent endpoint. Real QLever federation endpoints
// (`.../api/<dataset>`) have a second path segment / no trailing slash, so they
// are not matched.
inline bool looksLikeMagicServiceIri(std::string_view iri) {
  if (iri.size() < MAGIC_SERVICE_IRI_NAMESPACE.size() + 2 ||
      iri.substr(0, MAGIC_SERVICE_IRI_NAMESPACE.size()) !=
          MAGIC_SERVICE_IRI_NAMESPACE ||
      iri.back() != '>' || iri[iri.size() - 2] != '/') {
    return false;
  }
  std::string_view name =
      iri.substr(MAGIC_SERVICE_IRI_NAMESPACE.size(),
                 iri.size() - MAGIC_SERVICE_IRI_NAMESPACE.size() - 2);
  return !name.empty() && name.find_first_not_of(
                              "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
                              "WXYZ0123456789-") == std::string_view::npos;
}

// For backward compatibility: invocation of SpatialJoin via special predicates.
static const std::string MAX_DIST_IN_METERS = "<max-distance-in-meters:";
static const std::string NEAREST_NEIGHBORS = "<nearest-neighbors:";
static constexpr auto MAX_DIST_IN_METERS_REGEX =
    ctll::fixed_string{"<max-distance-in-meters:(?<dist>[0-9]+)>"};
static constexpr auto NEAREST_NEIGHBORS_REGEX = ctll::fixed_string{
    "<nearest-neighbors:(?<results>[0-9]+)(:(?<dist>[0-9]+))?>"};

#endif  // QLEVER_SRC_PARSER_MAGICSERVICEIRICONSTANTS_H
