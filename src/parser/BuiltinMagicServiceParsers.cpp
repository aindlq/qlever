// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "parser/BuiltinMagicServiceParsers.h"

#include <memory>
#include <mutex>

#include "backports/StartsWithAndEndsWith.h"
#include "global/Constants.h"
#include "parser/ExternalValuesQuery.h"
#include "parser/MagicServiceIriConstants.h"
#include "parser/MagicServiceRegistry.h"
#include "parser/NamedCachedResult.h"
#include "parser/PathQuery.h"
#include "parser/TextSearchQuery.h"

namespace parsedQuery {

// _____________________________________________________________________________
void registerBuiltinMagicServiceParsers() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    auto& registry = MagicServiceRegistry::get();

    registry.addExact(PATH_SEARCH_IRI, [](const MagicServiceRegistry::Iri&) {
      return std::make_shared<PathQuery>();
    });

    registry.addExact(TEXT_SEARCH_IRI, [](const MagicServiceRegistry::Iri&) {
      return std::make_shared<TextSearchQuery>();
    });

    // External values: an exact IRI or an IRI carrying options after a prefix.
    registry.add(
        [](std::string_view iri) {
          return iri == EXTERNAL_VALUES_IRI ||
                 ql::starts_with(iri, EXTERNAL_VALUES_IRI_PREFIX);
        },
        [](const MagicServiceRegistry::Iri& iri) {
          return std::make_shared<ExternalValuesQuery>(iri);
        });

    // Named cached result: the IRI content starts with a fixed prefix (the
    // matcher receives the bracketed representation, hence the leading `<`).
    registry.add(
        [](std::string_view iri) {
          return iri.size() > 1 &&
                 ql::starts_with(iri.substr(1), CACHED_RESULT_WITH_NAME_PREFIX);
        },
        [](const MagicServiceRegistry::Iri& iri) {
          return std::make_shared<NamedCachedResult>(iri);
        });
  });
}

}  // namespace parsedQuery
