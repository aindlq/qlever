// Copyright 2026, University of Freiburg,
// Chair of Algorithms and Data Structures.

#ifndef QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSUNDEFS_H
#define QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSUNDEFS_H

// Force-included as the first header of every QLever translation unit on
// Windows (GCC/Clang `-include`, MSVC `/FI`). It includes, exactly once, all
// Windows headers that define macros colliding with SPARQL token names in
// the generated ANTLR parser (`DELETE`, `OPTIONAL`, `IN`) and with QLever
// identifiers (`JoinType::OPTIONAL`, `SubtreePlan::OPTIONAL`,
// `S2BooleanOperation::OpType::DIFFERENCE`, antlr4's `ATNSimulator::ERROR`).
// Because all these headers have include guards, no later transitive
// include (Boost.Asio -> winsock2.h, Boost.UUID -> bcrypt.h, ...) can ever
// re-introduce the macros, so the #undefs below hold for the entire TU.
//
// This is the established pattern for the problem, cf. wxWidgets'
// `wx/msw/wrapwin.h`, Qt's `qt_windows.h`, and protobuf's `port_def.inc`
// (which additionally restores the macros via push/pop_macro - not usable
// here because QLever's own code uses the colliding names throughout).
// NOTE: This prelude is applied to QLever's own targets only, NOT to
// third-party dependency targets (they must see the standard macro
// environment; e.g. abseil's stacktrace implementation uses IN/OUT).
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef NOGDI
#define NOGDI
#endif
// clang-format off
#include <winsock2.h>  // must precede windows.h (Boost.Asio rule)
#include <ws2tcpip.h>
#include <windows.h>
#include <bcrypt.h>    // re-defines OPTIONAL/IN/OUT under #ifndef
// clang-format on
#undef DELETE    // winnt.h access right; no opt-out macro exists
#undef OPTIONAL  // minwindef.h/bcrypt.h pseudo-modifier
#undef IN        // minwindef.h/bcrypt.h pseudo-modifier
#undef OUT       // minwindef.h/bcrypt.h pseudo-modifier
#ifdef ERROR
#undef ERROR  // wingdi.h (prevented by NOGDI; belt and braces)
#endif
#ifdef DIFFERENCE
#undef DIFFERENCE  // winuser.h; collides with S2BooleanOperation::OpType
#endif
#ifdef IGNORE
#undef IGNORE
#endif
#undef near
#undef far
#ifdef min
#undef min
#endif
#ifdef max
#undef max
#endif
#ifdef interface
#undef interface
#endif

#endif  // _WIN32

#endif  // QLEVER_SRC_UTIL_SYS_WINDOWS_WINDOWSUNDEFS_H
