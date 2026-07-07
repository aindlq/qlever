# Building QLever on Windows

QLever has a native Windows port (experimental). It builds with **MinGW-w64
GCC** (MSVC is not supported) and passes the full unit-test suite and the
end-to-end tests. To stay in lockstep with the Linux CI it uses a **pinned**
toolchain — the same GCC and Boost versions as Linux, not a rolling release.

## Build

Reference toolchain (pin fixed versions):

- **Compiler:** [winlibs](https://winlibs.com/) GCC 13.3.0 (UCRT, POSIX threads,
  SEH) — bundles a matching CMake + Ninja.
- **Libraries via [Conan](https://conan.io/):** ICU, OpenSSL, zstd, zlib, bzip2,
  Boost 1.83 (`url`, `iostreams`, `program_options`, `container`; `stacktrace`
  excluded), and jemalloc 5.3.0 (`prefix=je_`; see the table below). The Conan
  Boost recipe pulls a newer b2 that already carries the one-line MinGW `gcc.jam`
  archive fix, so the only profile requirement is absolute compiler paths.

With the toolchain on `PATH`:

```bash
git config --global core.autocrlf false   # CRLF corrupts binary test data
git clone https://github.com/ad-freiburg/qlever.git && cd qlever
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/path/to/conan_toolchain.cmake
cmake --build build -j 6
ctest --test-dir build -j 8
```

No Windows-specific CMake flags are needed — everything is wired up
automatically when `WIN32` is detected.

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

All Windows code is isolated in **`src/util/windows/`** and **`windows/patches/`**,
plus small `#ifdef _WIN32` islands at the call sites. `WindowsUndefs.h` is
force-included (`-include`) into every translation unit to pull in `windows.h`
once and `#undef` the macros that collide with SPARQL/QLever identifiers
(`DELETE`, `OPTIONAL`, `ERROR`, ...).

### Why the port needs Windows-specific changes

| Area | Problem on Windows/MinGW | Fix |
|---|---|---|
| Positioned reads | `ReadFile` moves the handle's position; no POSIX `pread`. | `PositionedReadHandle` (`ReadFile` + `OVERLAPPED` offset on a dedicated handle). |
| Binary mode | CRT opens files in text mode → inserts/strips `\r`, treats `0x1A` as EOF → silent binary corruption. | `_set_fmode(_O_BINARY)` process-wide (`ProcessInit.cpp`) + explicit `_O_BINARY`. |
| Unlink/rename open files | `fopen` omits `FILE_SHARE_DELETE`. | `openWithPosixSharing` (`CreateFileA`, full sharing). |
| Rewrite a mapped file | A memory-mapped file can't be rewritten in place; `remove` only marks it delete-pending (name stays reserved) → recreate fails. | `posixDelete` (`SetFileInformationByHandle`, POSIX-semantics) frees the name immediately; `MmapVector` mappings use `FILE_SHARE_DELETE`. |
| `std::shared_mutex` | libstdc++'s is backed by winpthreads' lazily-initialized `pthread_rwlock`, which races on cold concurrent first use ([mingw-w64 #883](https://sourceforge.net/p/mingw-w64/bugs/883/)) → silent loss of mutual exclusion. | `Synchronized` uses a zero-init atomic shared mutex; `ConcurrentCache` uses a plain `std::mutex`; the ANTLR runtime is patched to eager-init. |
| Coarse sleeps | The ~15.6 ms timer tick makes `sleep_for` too coarse for timeouts/watchdogs. | `--wrap nanosleep64` → high-resolution waitable timer + microsecond residual spin, so `elapsed >= requested`. |
| Coroutine codegen | GCC-13/MinGW miscompiles SEH unwind tables for some C++20 coroutine frames → crash on unwind (GCC PR 101736/103274). | `-fno-reorder-blocks-and-partition` build-wide. |
| Open-file limit | The 512-`FILE*` CRT cap breaks the billion-triple vocabulary merge. | `_setmaxstdio(8192)`. |
| mmap vectors | No POSIX `mmap`; a mapped file can't be resized. | `boost::interprocess` mapping; `MmapVector` unmaps before resizing. |
| jemalloc | jemalloc can't interpose the CRT `malloc` on MinGW (its C API is `je_`-prefixed by design). | Pulled via Conan with `prefix=je_`; linking the static archive with `-static-libstdc++` routes C++ `new`/`delete` (where QLever's allocation traffic is) through jemalloc, while C `malloc` stays on the UCRT heap — so the shipped binary is fully static. |

### Vocabulary reads (performance)

The on-disk vocabulary is read with a memory-mapped `memcpy`
(`File::enableMemoryMappedReads`) instead of a per-call `ReadFile`. A warm
cached `ReadFile` still pays a structural kernel round-trip (syscall +
file-object lock + FCB resource + cached-read copy) — ~0.85 µs per call on the
vocabulary's ascending-offset scan (up to ~4 µs for fully random access) —
whereas the mapped `memcpy` is ~tens of nanoseconds. On an `on-disk-compressed`
vocabulary this is a **~6.5× speedup** on string-heavy queries (measured on
DBLP, 1.6 B triples; no effect with an in-memory vocab). The same effect holds
on Linux (mmap vs `pread`, ~2.6×), which is a cross-platform opportunity.
Benchmark numbers are in the PR description.

### Patched dependencies

`FetchContent` applies small **Windows-only** portability patches (fsst,
spatialjoin + its `util` submodule, the ANTLR runtime) — details in
[`windows/patches/README.md`](windows/patches/README.md). All are submitted
upstream and droppable once the pinned versions carry the fixes. The minimum
abseil/s2/re2 versions pinned in `CMakeLists.txt` also carry the needed MinGW
fixes and are required on every platform.

## Limitations

- **MinGW-w64 (GCC) only** — MSVC not attempted.
- **Indices are portable only across same-architecture platforms** (x86-64,
  little-endian): a Linux-built index loads and serves on Windows (verified
  end-to-end on a 1.58 B index); not across architectures or endianness.
- **Non-ASCII index paths** need routing through Boost.Nowide (not yet done) —
  keep index paths ASCII on Windows.
