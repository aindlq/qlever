// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_ENGINE_BUILTINMAGICSERVICEPLANNERS_H
#define QLEVER_SRC_ENGINE_BUILTINMAGICSERVICEPLANNERS_H

namespace qlever {

// Register the planner handlers of the built-in magic services that have been
// migrated onto the `MagicServicePlannerRegistry` (path search, text search,
// external values, named cached results). Idempotent (guarded by
// `std::call_once`); called from the planner before it dispatches a
// `MagicService` node, so the handlers are available in every binary without
// relying on static-initializer linkage.
void registerBuiltinMagicServicePlanners();

}  // namespace qlever

#endif  // QLEVER_SRC_ENGINE_BUILTINMAGICSERVICEPLANNERS_H
