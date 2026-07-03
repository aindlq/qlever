# Windows portability patches for fetched dependencies

These patches contain small portability fixes for dependencies that QLever
builds from source via `FetchContent`. They are applied automatically by the
patch step of the corresponding `FetchContent_Declare` in the top-level
`CMakeLists.txt`, and **only on Windows**, so that builds on other platforms
are completely unaffected. `ApplyPatch.cmake` makes the application
idempotent (re-running the patch step on an already-patched tree is a no-op).

Each patch can be deleted here once the pinned `GIT_TAG` of the dependency
contains the corresponding fix (the I/O patches have been submitted upstream;
the ANTLR one is a candidate to upstream).

## `fsst-windows.patch` (for cwida/fsst)

`fsst_avx512.cpp` guards an MSVC-only code path with `#ifdef _WIN32`, which
also (incorrectly) matches MinGW/GCC on Windows. The patch changes the guard
to `#if defined(_MSC_VER)`.

## `spatialjoin-windows.patch` (for ad-freiburg/spatialjoin)

Binary-mode file I/O for Windows: `open()` calls gain `O_BINARY` (defined to
0 on platforms that don't have it, so POSIX builds are unchanged), and
`fsync` is mapped to `_commit` on Windows. Without this, the Windows
text-mode default corrupts the binary intermediate files that spatialjoin
writes (CR insertion, 0x1A treated as EOF).

## `spatialjoin-util-windows.patch` (for ad-freiburg/util, a git submodule of spatialjoin)

The same kind of fixes for the `util` submodule of spatialjoin
(`src/util`): a guarded `pwd.h` include, `pread`/`pwrite` emulation, and a
`getHomeDir` that uses `USERPROFILE`. Applied with `git -C src/util apply`
because the submodule is a separate git repository.

## `antlr-windows.patch` (for antlr/antlr4, the C++ runtime)

Eager-initializes `antlr4::internal::SharedMutex`'s underlying
`std::shared_mutex` (a single-threaded `lock()`/`unlock()` in the
constructor). On MinGW, libstdc++'s `std::shared_mutex` is backed by
winpthreads' `pthread_rwlock`, which is lazily initialized on first lock and
races on cold concurrent first use (mingw-w64 bug
[#883](https://sourceforge.net/p/mingw-w64/bugs/883/)) — the init-race loser
gets `EINVAL`, libstdc++ ignores it in release builds, and two threads then
enter the "exclusive" section at once, silently losing mutual exclusion.
QLever's own `ThreadAgnosticSharedMutex` does not cover ANTLR, which uses
`std::shared_mutex` on the SPARQL parsing path. Forcing eager construction of
the rwlock (single-threaded, before the mutex is shared) closes the race.
Only the `std::shared_mutex` branch is patched (`#else` of
`ANTLR4CPP_USING_ABSEIL`); the Abseil branch is unaffected.

