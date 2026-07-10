// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem Kozlov <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_UTIL_SYS_ACCESSPATTERN_H
#define QLEVER_SRC_UTIL_SYS_ACCESSPATTERN_H

namespace ad_utility {
// madvise-style access-pattern hint for a memory mapping.
enum class AccessPattern { None, Random, Sequential };
}  // namespace ad_utility

#endif  // QLEVER_SRC_UTIL_SYS_ACCESSPATTERN_H
