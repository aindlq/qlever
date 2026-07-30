# Building QLever on Windows

QLever has a native Windows port (experimental). It builds with **MinGW-w64
GCC** (MSVC is not supported) and passes the full unit-test suite and the
end-to-end tests. It uses a **pinned** toolchain rather than rolling dependency
versions. The GCC major stays aligned with Linux CI; Boost is newer on Windows
to include fixes in Boost.Asio's thread-pool teardown.

## Build

Reference toolchain:

- **Compiler:** [winlibs](https://winlibs.com/) GCC 13.3.0 (UCRT, POSIX threads,
  SEH) — bundles a matching CMake + Ninja.
- **Libraries via [Conan](https://conan.io/):** ICU, OpenSSL, zstd, zlib, bzip2,
  Boost 1.88 (`url`, `iostreams`, `program_options`, `container`; `stacktrace`
  excluded), and jemalloc 5.3.0 (`prefix=je_`; see the table below). The Conan
  Boost recipe pulls a newer b2 that already carries the one-line MinGW `gcc.jam`
  archive fix, so the only profile requirement is absolute compiler paths.
- **Build tools:** Python, Conan 2.29.1, and MSYS2 at `C:/msys64` with `make`,
  Perl, Autoconf, Automake, Libtool, `tar`, and `gawk`.

Install winlibs at `C:/winlibs13` (or set `QLEVER_WINLIBS_DIR`), put its
`mingw64/bin` directory on `PATH`, and run the following from a Bash shell. Set
`QLEVER_MSYS2_DIR` too if MSYS2 is not installed at `C:/msys64`.

```bash
git config --global core.autocrlf false   # CRLF corrupts binary test data
git clone https://github.com/ad-freiburg/qlever.git && cd qlever

export QLEVER_WINLIBS_DIR=C:/winlibs13
export QLEVER_MSYS2_DIR=C:/msys64
C:/msys64/usr/bin/pacman.exe -S --noconfirm --needed \
    make perl autoconf automake libtool tar gawk
python -m pip install conan==2.29.1
conan install conanprofiles/conanfile-windows-deps.txt \
    -pr:a conanprofiles/gcc-13-windows \
    --lockfile=conanprofiles/windows.lock \
    --output-folder=build --build=missing

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE="$PWD/build/conan_toolchain.cmake"
cmake --build build -j 6
ctest --test-dir build --no-tests=error -j 8
```

This produces a normal dynamically linked developer build. For the
self-contained artifact configuration used by CI, add
`-DSTATIC_LINKING=ON -DQLEVER_REQUIRE_JEMALLOC=ON` to the CMake command.
The Conan lockfile pins recipe revisions and transitive versions; update it
deliberately when changing the dependency set.

> **Static builds silently disable the parallel index sort.** QLever's
> `USE_PARALLEL` (the parallel permutation sort in the index builder) only takes
> effect if CMake's `FindOpenMP` succeeds. If you link statically for a
> self-contained binary (`-DCMAKE_EXE_LINKER_FLAGS="-static -static-libgcc
> -static-libstdc++"`), FindOpenMP's test-link fails on MinGW — it can't resolve
> libgomp under `-static` — so `USE_PARALLEL` degrades to a serial `std::sort`
> with no error. The result is a **~1.5× slower index build** (measured 3971 s
> serial vs 2724 s parallel on 1.6 B triples; query performance is unaffected).
> To keep the parallel sort in a static build, force the OpenMP variables past
> detection and link a no-op `dl*` stub so static libgomp's `dlopen` resolves:
>
> ```bash
> printf 'void*dlopen(const char*f,int m){return 0;}\nvoid*dlsym(void*h,const char*s){return 0;}\nint dlclose(void*h){return 0;}\nconst char*dlerror(void){return 0;}\n' > dl_stub.c
> gcc -c dl_stub.c -o dl_stub.o
> cmake -B build ... \
>     -DOpenMP_CXX_FLAGS=-fopenmp -DOpenMP_CXX_LIB_NAMES=gomp \
>     -DOpenMP_gomp_LIBRARY=/path/to/mingw64/lib/libgomp.a \
>     -DCMAKE_CXX_STANDARD_LIBRARIES="$PWD/dl_stub.o"
> ```
>
> A dynamic build (no `-static`) detects OpenMP normally and needs none of this.

## Port architecture

All Windows code is isolated in **`src/util/sys/windows/`** and **`windows/patches/`**,
plus small `#ifdef _WIN32` islands at the call sites. `WindowsUndefs.h` is
force-included (`-include`) into every translation unit to pull in `windows.h`
once and `#undef` the macros that collide with SPARQL/QLever identifiers
(`DELETE`, `OPTIONAL`, `ERROR`, ...).

### Why the port needs Windows-specific changes

| Area | Problem on Windows/MinGW | Fix |
|---|---|---|
| Positioned reads | `ReadFile` moves the handle's position; no POSIX `pread`. | `PositionedReadHandle` (`ReadFile` + `OVERLAPPED` offset on dedicated handles, derived from the fd via `ReOpenFile` so reads survive renames like `pread` after `dup`). |
| Binary mode | CRT opens files in text mode → inserts/strips `\r`, treats `0x1A` as EOF → silent binary corruption. | `_set_fmode(_O_BINARY)` process-wide (`ProcessInit.cpp`) + explicit `_O_BINARY`. |
| Unlink/rename open files | `fopen` omits `FILE_SHARE_DELETE`. | `openWithPosixSharing` (`CreateFileA`, full sharing). |
| Replace an open file | Plain Windows deletion can leave a name delete-pending while handles are open, blocking immediate recreation. | `posixDelete` (`SetFileInformationByHandle`, POSIX semantics) and `FILE_SHARE_DELETE` provide the unlink-like behavior QLever expects. |
| `std::shared_mutex` | libstdc++'s is backed by winpthreads' lazily initialized `pthread_rwlock`, which can lose mutual exclusion on concurrent first use ([mingw-w64 #883](https://sourceforge.net/p/mingw-w64/bugs/883/)) and leak its backing allocation on destruction ([#1012](https://sourceforge.net/p/mingw-w64/bugs/1012/)). | QLever's synchronized state uses `std::mutex`; read-only access remains exclusive. |
| Coarse sleeps | The ~15.6 ms timer tick makes `sleep_for` too coarse for timeouts/watchdogs. | `--wrap nanosleep64` → high-resolution waitable timer + microsecond residual spin, so `elapsed >= requested`. |
| Coroutine codegen | GCC-13/MinGW miscompiles SEH unwind tables for some C++20 coroutine frames → crash on unwind (GCC PR 101736/103274). | `-fno-reorder-blocks-and-partition` build-wide. |
| Open-file limit | The 512-`FILE*` CRT cap breaks the billion-triple vocabulary merge. | `_setmaxstdio(8192)`. |
| jemalloc | jemalloc can't interpose the CRT `malloc` on MinGW (its C API is `je_`-prefixed by design). | Pulled via Conan with `prefix=je_`; the static artifact build places its archive before a fully static libstdc++, routing C++ `new`/`delete` through jemalloc while C `malloc` stays on the UCRT heap. |

### Vocabulary reads

QLever's batched vocabulary lookups use the upstream `BatchManager` design.
Linux uses `io_uring` when available and otherwise falls back to synchronous
`pread`. On Windows the same fallback is backed by positioned `ReadFile` calls
on dedicated handles obtained with `ReOpenFile`; this preserves `pread`
semantics and avoids memory mapping.

### Patched dependencies

`FetchContent` applies small **Windows-only** portability patches (fsst,
spatialjoin + its `util` submodule) — details and upstream
status are recorded in [`windows/patches/README.md`](windows/patches/README.md).
Each patch is droppable once the pinned dependency contains its fix. The
minimum abseil/s2/re2 versions pinned in `CMakeLists.txt` also carry the needed
MinGW fixes and are required on every platform.

## Limitations

- **MinGW-w64 (GCC) only** — MSVC not attempted.
- **The C++17 backports configuration is not supported** — the Windows Conan
  package deliberately omits Boost.Filesystem; use QLever's default C++20 mode.
- **Indices are portable only across same-architecture platforms** (x86-64,
  little-endian): a Linux-built index loads and serves on Windows (verified
  end-to-end on a 1.58 B index); not across architectures or endianness.
- **Non-ASCII index paths** need routing through Boost.Nowide (not yet done) —
  keep index paths ASCII on Windows.
