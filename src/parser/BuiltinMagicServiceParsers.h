// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_PARSER_BUILTINMAGICSERVICEPARSERS_H
#define QLEVER_SRC_PARSER_BUILTINMAGICSERVICEPARSERS_H

namespace parsedQuery {

// Register the parser factories of the built-in magic services that have been
// migrated onto the `MagicServiceRegistry` (path search, text search, external
// values, named cached results). Idempotent (guarded by `std::call_once`);
// called from the SPARQL visitor before it dispatches a `SERVICE` IRI, so the
// factories are available in every binary without relying on
// static-initializer linkage.
void registerBuiltinMagicServiceParsers();

}  // namespace parsedQuery

#endif  // QLEVER_SRC_PARSER_BUILTINMAGICSERVICEPARSERS_H
