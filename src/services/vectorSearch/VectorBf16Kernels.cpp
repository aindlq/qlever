// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorBf16Kernels.h"

#include <cstring>

#if defined(__linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

// Per-function `target` attributes (below) enable AMX / AVX-512-BF16 for the
// hot functions WITHOUT a global `-march`, so this TU stays portable and the
// SIMD-off / non-x86 build still compiles (the functions become inert stubs
// and the `*Available()` probes return false). This mirrors NumKong's own
// per-ISA kernel isolation.

#if defined(__x86_64__) || defined(_M_X64)
#define QL_VEC_BF16_X86 1
#include <immintrin.h>
#include <cpuid.h>
#else
#define QL_VEC_BF16_X86 0
#endif

namespace qlever::vector::bf16kernels {

#if QL_VEC_BF16_X86

// ---- CPU capability probes (cached) -------------------------------------
namespace {
// CPUID(7,0).EDX bit 22 = AMX-BF16; bit 24 = AMX-TILE.
// CPUID(7,1).EAX bit 5  = AVX512-BF16.
// CPUID(7,0).EBX bit 16 = AVX512F, bit 30 = AVX512BW, bit 17 = AVX512DQ,
//                bit 31 = AVX512VL.
struct CpuProbe {
  bool simd = false;  // AVX-512-BF16 (+ f/bw/dq/vl)
  bool amx = false;   // AMX-TILE + AMX-BF16
  CpuProbe() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
      return;
    }
    const bool avx512f = (b >> 16) & 1u;
    const bool avx512bw = (b >> 30) & 1u;
    const bool avx512vl = (b >> 31) & 1u;
    const bool amxTile = (d >> 24) & 1u;
    const bool amxBf16 = (d >> 22) & 1u;
    unsigned a1, b1, c1, d1;
    bool avx512bf16 = false;
    if (__get_cpuid_count(7, 1, &a1, &b1, &c1, &d1)) {
      avx512bf16 = (a1 >> 5) & 1u;
    }
    const bool baseAvx512 = avx512f && avx512bw && avx512vl;
    simd = baseAvx512 && avx512bf16;
    amx = simd && amxTile && amxBf16;
  }
};
const CpuProbe& probe() {
  static const CpuProbe p;
  return p;
}
}  // namespace

bool simdAvailable() { return probe().simd; }
bool amxAvailable() { return probe().amx; }

bool ensureAmxPermission() {
  if (!probe().amx) {
    return false;
  }
#if defined(__linux__)
  // Cache the (idempotent) request + verify: the permission is process-wide.
  static const bool granted = [] {
    constexpr int kArchReqXcompPerm = 0x1023;  // ARCH_REQ_XCOMP_PERM
    constexpr int kArchGetXcompPerm = 0x1022;  // ARCH_GET_XCOMP_PERM
    constexpr unsigned long kXtiledata = 18;   // XFEATURE_XTILEDATA
    // Request. A zero return means the kernel accepted (or already had) it.
    if (syscall(SYS_arch_prctl, kArchReqXcompPerm, kXtiledata) != 0) {
      return false;
    }
    // Verify: read the granted feature bitmap back and check the tile-data bit
    // is actually set (a kernel/policy may accept the call yet not grant it).
    unsigned long bitmap = 0;
    if (syscall(SYS_arch_prctl, kArchGetXcompPerm, &bitmap) != 0) {
      return false;
    }
    return (bitmap & (1UL << kXtiledata)) != 0;
  }();
  return granted;
#else
  // Non-Linux (Windows/FreeBSD): AMX tile state is enabled by the OS without an
  // explicit request. Trust the CPUID probe.
  return true;
#endif
}

// ---- SIMD (AVX-512-BF16) ------------------------------------------------

// One row's dot against the (register-resident) query, `dim` a multiple of 32.
// `q` and `r` are bf16 element pointers. Kept small so the multi-row driver
// can inline and keep 8 accumulators live.
__attribute__((target("avx512bf16,avx512vl,avx512bw,avx512f"))) static inline float
dotOneSimd_(const uint16_t* q, const uint16_t* r, size_t dim) {
  __m512 acc = _mm512_setzero_ps();
  for (size_t c = 0; c < dim; c += 32) {
    __m512i qq = _mm512_loadu_si512(reinterpret_cast<const void*>(q + c));
    __m512i rr = _mm512_loadu_si512(reinterpret_cast<const void*>(r + c));
    acc = _mm512_dpbf16_ps(acc, reinterpret_cast<__m512bh>(qq),
                           reinterpret_cast<__m512bh>(rr));
  }
  return _mm512_reduce_add_ps(acc);
}

// 8 rows per iteration, 8 independent `vdpbf16ps` accumulator chains, query
// reloaded from L1 each 64 B chunk -- the `dot8` prototype. `rowPtr(j)` yields
// the bf16 element pointer of output row j (contiguous or gathered).
template <typename RowPtrFn>
__attribute__((target("avx512bf16,avx512vl,avx512bw,avx512f"))) static void
dotMultiSimd_(const uint16_t* q, RowPtrFn rowPtr, size_t count, size_t dim,
              float* out) {
  size_t j = 0;
  for (; j + 8 <= count; j += 8) {
    const uint16_t* r0 = rowPtr(j + 0);
    const uint16_t* r1 = rowPtr(j + 1);
    const uint16_t* r2 = rowPtr(j + 2);
    const uint16_t* r3 = rowPtr(j + 3);
    const uint16_t* r4 = rowPtr(j + 4);
    const uint16_t* r5 = rowPtr(j + 5);
    const uint16_t* r6 = rowPtr(j + 6);
    const uint16_t* r7 = rowPtr(j + 7);
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0,
           a5 = a0, a6 = a0, a7 = a0;
    for (size_t c = 0; c < dim; c += 32) {
      __m512bh qv = reinterpret_cast<__m512bh>(
          _mm512_loadu_si512(reinterpret_cast<const void*>(q + c)));
#define QL_STEP(idx, acc, rp)                                              \
  acc = _mm512_dpbf16_ps(acc, qv,                                         \
                         reinterpret_cast<__m512bh>(_mm512_loadu_si512(   \
                             reinterpret_cast<const void*>((rp) + c))))
      QL_STEP(0, a0, r0);
      QL_STEP(1, a1, r1);
      QL_STEP(2, a2, r2);
      QL_STEP(3, a3, r3);
      QL_STEP(4, a4, r4);
      QL_STEP(5, a5, r5);
      QL_STEP(6, a6, r6);
      QL_STEP(7, a7, r7);
#undef QL_STEP
    }
    out[j + 0] = _mm512_reduce_add_ps(a0);
    out[j + 1] = _mm512_reduce_add_ps(a1);
    out[j + 2] = _mm512_reduce_add_ps(a2);
    out[j + 3] = _mm512_reduce_add_ps(a3);
    out[j + 4] = _mm512_reduce_add_ps(a4);
    out[j + 5] = _mm512_reduce_add_ps(a5);
    out[j + 6] = _mm512_reduce_add_ps(a6);
    out[j + 7] = _mm512_reduce_add_ps(a7);
  }
  for (; j < count; ++j) {
    out[j] = dotOneSimd_(q, rowPtr(j), dim);
  }
}

void dotBlockSimd(const uint16_t* query, const char* rows,
                  size_t rowStrideBytes, size_t count, size_t dim, float* out) {
  dotMultiSimd_(
      query,
      [&](size_t j) {
        return reinterpret_cast<const uint16_t*>(rows + j * rowStrideBytes);
      },
      count, dim, out);
}

void dotGatherSimd(const uint16_t* query, const char* base,
                   size_t rowStrideBytes, const size_t* rowOf, size_t count,
                   size_t dim, float* out) {
  dotMultiSimd_(
      query,
      [&](size_t j) {
        return reinterpret_cast<const uint16_t*>(base +
                                                 rowOf[j] * rowStrideBytes);
      },
      count, dim, out);
}

float dotPairSimd(const uint16_t* a, const uint16_t* b, size_t dim) {
  return dotOneSimd_(a, b, dim);
}

// ---- AMX (fixed width-1 tile GEMM) --------------------------------------

// One B tile: 16 depth-groups x 16 columns x 2 (pair-interleaved) bf16 = 1 KB.
// Only column 0 is non-zero for a width-1 query.
namespace {
constexpr size_t AMX_TILE_BYTES = 1024;
constexpr size_t AMX_DEPTH = 32;  // bf16 elements per depth tile

// Per-thread AMX tile configuration (palette 1, eight 16x64B tiles). Cheap;
// done once per `dotBlockAmx` call. Carries the SAME target set as
// `dotBlockAmx` (a strict superset of the `amx-tile` it needs) so the compiler
// may freely inline it into the caller -- no target-subset mismatch that would
// force it out-of-line or, worse, break the build if it were marked inline.
__attribute__((target("amx-tile,amx-bf16,avx512bf16,avx512vl,avx512bw,avx512f")))
void amxConfigure_() {
  alignas(64) uint8_t cfg[64] = {0};
  cfg[0] = 1;  // palette 1
  uint16_t* bytesPerRow = reinterpret_cast<uint16_t*>(&cfg[16]);
  uint8_t* rowsPerTile = &cfg[48];
  for (int i = 0; i < 8; ++i) {
    rowsPerTile[i] = 16;
    bytesPerRow[i] = 64;
  }
  _tile_loadconfig(cfg);
}
}  // namespace

size_t packedQuerySizeAmx(size_t dim) { return (dim / AMX_DEPTH) * AMX_TILE_BYTES; }

void packQueryAmx(const uint16_t* query, size_t dim, void* dst) {
  // B-tile layout: data[depthGroup][column][pair], 16 x 16 x 2 bf16. Element
  // (depth k, column col) lives at data[k/2][col][k%2]. Only column 0 is
  // populated; the rest stay zero (memset). No SIMD needed -- this runs ONCE
  // per search.
  const size_t depthTiles = dim / AMX_DEPTH;
  std::memset(dst, 0, depthTiles * AMX_TILE_BYTES);
  auto* tiles = reinterpret_cast<uint16_t*>(dst);
  for (size_t dt = 0; dt < depthTiles; ++dt) {
    uint16_t* tile = tiles + dt * (AMX_TILE_BYTES / sizeof(uint16_t));
    for (size_t k = 0; k < AMX_DEPTH; ++k) {
      // data[k/2][0][k%2] -> flat index ((k/2) * 16 + 0) * 2 + (k%2).
      const size_t flat = ((k / 2) * 16 + 0) * 2 + (k % 2);
      tile[flat] = query[dt * AMX_DEPTH + k];
    }
  }
}

// Contiguous 32-row block via AMX; the depth-tile loop pipelines A loads two
// deep. `count` need NOT be a multiple of 32 -- the tail (< 32 rows) is scored
// by the SIMD one-row kernel so every output is exact.
__attribute__((target("amx-tile,amx-bf16,avx512bf16,avx512vl,avx512bw,avx512f"))) void
dotBlockAmx(const void* packedQuery, const uint16_t* query, const char* rows,
            size_t rowStrideBytes, size_t count, size_t dim, float* out) {
  const size_t depthTiles = dim / AMX_DEPTH;
  const auto* btiles = reinterpret_cast<const char*>(packedQuery);
  amxConfigure_();
  alignas(64) float c0[16][16];
  alignas(64) float c1[16][16];
  size_t r = 0;
  for (; r + 32 <= count; r += 32) {
    const char* aTop = rows + r * rowStrideBytes;
    const char* aBot = aTop + 16 * rowStrideBytes;
    _tile_zero(4);
    _tile_zero(6);
    // Prologue: first depth tile.
    _tile_loadd(0, aTop, static_cast<int>(rowStrideBytes));
    _tile_loadd(1, aBot, static_cast<int>(rowStrideBytes));
    _tile_loadd(2, btiles, 64);
    for (size_t dt = 0; dt + 1 < depthTiles; ++dt) {
      _tile_dpbf16ps(4, 0, 2);
      _tile_dpbf16ps(6, 1, 2);
      _tile_loadd(0, aTop + (dt + 1) * 64, static_cast<int>(rowStrideBytes));
      _tile_loadd(1, aBot + (dt + 1) * 64, static_cast<int>(rowStrideBytes));
      _tile_loadd(2, btiles + (dt + 1) * AMX_TILE_BYTES, 64);
    }
    _tile_dpbf16ps(4, 0, 2);
    _tile_dpbf16ps(6, 1, 2);
    _tile_stored(4, c0, 64);
    _tile_stored(6, c1, 64);
    for (int i = 0; i < 16; ++i) {
      out[r + i] = c0[i][0];
      out[r + 16 + i] = c1[i][0];
    }
  }
  _tile_release();
  // Tail rows (< 32): exact via the SIMD one-row dot on the plain query, so
  // no output is left to a masked AMX edge tile (bit-identical to a full AMX
  // block for the same pair -- same bf16 dot).
  for (; r < count; ++r) {
    out[r] = dotOneSimd_(
        query, reinterpret_cast<const uint16_t*>(rows + r * rowStrideBytes),
        dim);
  }
}

#else  // !QL_VEC_BF16_X86 -- inert stubs (never selected; probes false).

bool simdAvailable() { return false; }
bool amxAvailable() { return false; }
bool ensureAmxPermission() { return false; }
void dotBlockSimd(const uint16_t*, const char*, size_t, size_t, size_t,
                  float*) {}
void dotGatherSimd(const uint16_t*, const char*, size_t, const size_t*, size_t,
                   size_t, float*) {}
float dotPairSimd(const uint16_t*, const uint16_t*, size_t) { return 0.f; }
size_t packedQuerySizeAmx(size_t) { return 0; }
void packQueryAmx(const uint16_t*, size_t, void*) {}
void dotBlockAmx(const void*, const uint16_t*, const char*, size_t, size_t,
                 size_t, float*) {}

#endif

}  // namespace qlever::vector::bf16kernels
