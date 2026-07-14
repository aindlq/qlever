// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#include "services/vectorSearch/VectorI8Kernels.h"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <string>

// Per-function `target` attributes (below) enable AVX-512-VNNI for the hot
// functions WITHOUT a global `-march`, so this TU stays portable and the
// SIMD-off / non-x86 build still compiles (the functions become inert stubs
// and the `vnniAvailable()` probe returns false) -- the same isolation as
// `VectorBf16Kernels.cpp`.

#if defined(__x86_64__) || defined(_M_X64)
#define QL_VEC_I8_X86 1
#include <cpuid.h>
#include <immintrin.h>
#else
#define QL_VEC_I8_X86 0
#endif

namespace qlever::vector::i8kernels {

// ---- portable exact helpers ----------------------------------------------

void rowSums(const int8_t* v, size_t dim, int32_t* sum, int32_t* normSq) {
  // Plain scalar: exact for the full i8 range (including -128), trivially
  // auto-vectorizable, and only run once per row at open / per query.
  int32_t s = 0, s2 = 0;
  for (size_t i = 0; i < dim; ++i) {
    const int32_t x = v[i];
    s += x;
    s2 += x * x;
  }
  *sum = s;
  *normSq = s2;
}

#if QL_VEC_I8_X86

// ---- CPU capability probe (cached) ---------------------------------------
namespace {
// CPUID(7,0).EBX bit 16 = AVX512F, bit 30 = AVX512BW, bit 31 = AVX512VL.
// CPUID(7,0).ECX bit 11 = AVX512-VNNI.
struct CpuProbe {
  bool vnni = false;
  CpuProbe() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
      return;
    }
    const bool avx512f = (b >> 16) & 1u;
    const bool avx512bw = (b >> 30) & 1u;
    const bool avx512vl = (b >> 31) & 1u;
    const bool avx512vnni = (c >> 11) & 1u;
    vnni = avx512f && avx512bw && avx512vl && avx512vnni;
  }
};
const CpuProbe& probe() {
  static const CpuProbe p;
  return p;
}

// A/B off-switch for the contiguous-sweep software prefetch, sharing the
// bf16 kernels' env knob (one process-wide policy). Memoized.
bool prefetchEnabled_() {
  static const bool on = [] {
    const char* v = std::getenv("QLEVER_VECTOR_SEARCH_PREFETCH");
    if (v == nullptr) return true;
    std::string s{v};
    for (char& c : s) c = static_cast<char>(std::tolower((unsigned char)c));
    return !(s == "0" || s == "false" || s == "off" || s == "no");
  }();
  return on;
}
}  // namespace

bool vnniAvailable() { return probe().vnni; }

// ---- the shared finalize (ONE pinned instance) ----------------------------

// EXACT structural replica of NumKong's `nk_angular_normalize_f32_haswell_`,
// written entirely in explicit SSE intrinsics (baseline x86-64; `_mm_rsqrt_ps`
// is SSE1) and marked NOINLINE, so there is exactly ONE compiled instance
// whose instruction sequence does not depend on the caller's ISA context or
// on `-ffp-contract` -- that is what makes every i8 path bit-identical to
// every other. (The punned metric's copy of the same source is compiled
// under NumKong's icelake context, whose FMA contraction can differ by
// ~1 ulp on a small fraction of pairs; see the header.)
__attribute__((noinline)) float angularFinalize(float dot, float a2, float b2) {
  if (a2 == 0.0f && b2 == 0.0f) return 0.0f;
  if (dot == 0.0f) return 1.0f;
  __m128 squares = _mm_set_ps(a2, b2, a2, b2);
  __m128 rsqrts = _mm_rsqrt_ps(squares);
  // One Newton-Raphson step: y' = y * (1.5 - 0.5 * x * y * y).
  __m128 half = _mm_set1_ps(0.5f);
  __m128 threeHalves = _mm_set1_ps(1.5f);
  rsqrts = _mm_mul_ps(
      rsqrts,
      _mm_sub_ps(
          threeHalves,
          _mm_mul_ps(half, _mm_mul_ps(squares, _mm_mul_ps(rsqrts, rsqrts)))));
  __m128 a2r = _mm_shuffle_ps(rsqrts, rsqrts, _MM_SHUFFLE(0, 0, 0, 1));
  __m128 b2r = rsqrts;
  // 1 - dot * a2r * b2r, evaluated as two explicit scalar multiplies and a
  // subtract (no contraction), then clamped at 0.
  __m128 prod = _mm_mul_ss(_mm_mul_ss(_mm_set_ss(dot), a2r), b2r);
  float result = 1.0f - _mm_cvtss_f32(prod);
  return result > 0.0f ? result : 0.0f;
}

// ---- one-row exact signed dot (vpmaddwd) ----------------------------------

// 32 elements per step: sign-extend i8 -> i16, `vpmaddwd` into 16 i32 lanes.
// Exact for the full i8 range (products fit i16*i16 -> i32 pairs).
__attribute__((target("avx512f,avx512bw,avx512vl,bmi2"))) int32_t
dotPair(const int8_t* a, const int8_t* b, size_t dim) {
  __m512i acc = _mm512_setzero_si512();
  size_t c = 0;
  for (; c + 32 <= dim; c += 32) {
    __m512i av = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a + c)));
    __m512i bv = _mm512_cvtepi8_epi16(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + c)));
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(av, bv));
  }
  if (c < dim) {
    const __mmask32 mask =
        static_cast<__mmask32>(_bzhi_u32(0xFFFFFFFFu, (unsigned)(dim - c)));
    __m512i av = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, a + c));
    __m512i bv = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(mask, b + c));
    acc = _mm512_add_epi32(acc, _mm512_madd_epi16(av, bv));
  }
  return _mm512_reduce_add_epi32(acc);
}

// ---- multi-row VNNI block / gather ----------------------------------------

namespace {
// One row's biased dot, for tails of the 8-row driver.
__attribute__((
    target("avx512f,avx512bw,avx512vl,avx512vnni,bmi2"))) inline int32_t
dotOneVnni_(const uint8_t* qx, const int8_t* r, size_t dim) {
  __m512i acc = _mm512_setzero_si512();
  size_t c = 0;
  for (; c + 64 <= dim; c += 64) {
    __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void*>(qx + c));
    __m512i rv = _mm512_loadu_si512(reinterpret_cast<const void*>(r + c));
    acc = _mm512_dpbusd_epi32(acc, qv, rv);
  }
  if (c < dim) {
    const __mmask64 mask = _bzhi_u64(~0ull, (unsigned)(dim - c));
    __m512i qv = _mm512_maskz_loadu_epi8(mask, qx + c);
    __m512i rv = _mm512_maskz_loadu_epi8(mask, r + c);
    acc = _mm512_dpbusd_epi32(acc, qv, rv);
  }
  return _mm512_reduce_add_epi32(acc);
}

// 8 rows per iteration, 8 independent `vpdpbusd` accumulator chains -- the
// i8 analog of the bf16 `dot8` driver. `rowPtr(j)` yields the i8 pointer of
// output row j (contiguous or gathered). `Prefetch` (contiguous only): one
// T2 prefetch per consumed 64 B line, each row's own stream `pfBytes` ahead.
template <bool Prefetch, typename RowPtrFn>
__attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni,bmi2"))) void
dotMultiVnni_(const uint8_t* qx, RowPtrFn rowPtr, size_t count, size_t dim,
              int32_t* out, size_t pfBytes) {
  size_t j = 0;
  for (; j + 8 <= count; j += 8) {
    const int8_t* r0 = rowPtr(j + 0);
    const int8_t* r1 = rowPtr(j + 1);
    const int8_t* r2 = rowPtr(j + 2);
    const int8_t* r3 = rowPtr(j + 3);
    const int8_t* r4 = rowPtr(j + 4);
    const int8_t* r5 = rowPtr(j + 5);
    const int8_t* r6 = rowPtr(j + 6);
    const int8_t* r7 = rowPtr(j + 7);
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0,
            a5 = a0, a6 = a0, a7 = a0;
    size_t c = 0;
    for (; c + 64 <= dim; c += 64) {
      __m512i qv = _mm512_loadu_si512(reinterpret_cast<const void*>(qx + c));
      if constexpr (Prefetch) {
        _mm_prefetch(reinterpret_cast<const char*>(r0) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r1) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r2) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r3) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r4) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r5) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r6) + c + pfBytes,
                     _MM_HINT_T2);
        _mm_prefetch(reinterpret_cast<const char*>(r7) + c + pfBytes,
                     _MM_HINT_T2);
      }
#define QL_STEP(acc, rp)     \
  acc = _mm512_dpbusd_epi32( \
      acc, qv, _mm512_loadu_si512(reinterpret_cast<const void*>((rp) + c)))
      QL_STEP(a0, r0);
      QL_STEP(a1, r1);
      QL_STEP(a2, r2);
      QL_STEP(a3, r3);
      QL_STEP(a4, r4);
      QL_STEP(a5, r5);
      QL_STEP(a6, r6);
      QL_STEP(a7, r7);
#undef QL_STEP
    }
    if (c < dim) {
      const __mmask64 mask = _bzhi_u64(~0ull, (unsigned)(dim - c));
      __m512i qv = _mm512_maskz_loadu_epi8(mask, qx + c);
#define QL_TAIL(acc, rp) \
  acc = _mm512_dpbusd_epi32(acc, qv, _mm512_maskz_loadu_epi8(mask, (rp) + c))
      QL_TAIL(a0, r0);
      QL_TAIL(a1, r1);
      QL_TAIL(a2, r2);
      QL_TAIL(a3, r3);
      QL_TAIL(a4, r4);
      QL_TAIL(a5, r5);
      QL_TAIL(a6, r6);
      QL_TAIL(a7, r7);
#undef QL_TAIL
    }
    out[j + 0] = _mm512_reduce_add_epi32(a0);
    out[j + 1] = _mm512_reduce_add_epi32(a1);
    out[j + 2] = _mm512_reduce_add_epi32(a2);
    out[j + 3] = _mm512_reduce_add_epi32(a3);
    out[j + 4] = _mm512_reduce_add_epi32(a4);
    out[j + 5] = _mm512_reduce_add_epi32(a5);
    out[j + 6] = _mm512_reduce_add_epi32(a6);
    out[j + 7] = _mm512_reduce_add_epi32(a7);
  }
  for (; j < count; ++j) {
    out[j] = dotOneVnni_(qx, rowPtr(j), dim);
  }
}
}  // namespace

void dotBlockVnni(const uint8_t* qx, const char* rows, size_t rowStrideBytes,
                  size_t count, size_t dim, int32_t* out) {
  auto rowPtr = [&](size_t j) {
    return reinterpret_cast<const int8_t*>(rows + j * rowStrideBytes);
  };
  // Prefetch 4 rows ahead (~4.6 KB at the 1152 B image rows) -- the same
  // tuned distance as the bf16 block. Two compiled variants; the memoized
  // env flag picks one, no per-row branch.
  if (prefetchEnabled_()) {
    dotMultiVnni_<true>(qx, rowPtr, count, dim, out, 4 * rowStrideBytes);
  } else {
    dotMultiVnni_<false>(qx, rowPtr, count, dim, out, 0);
  }
}

void dotGatherVnni(const uint8_t* qx, const char* base, size_t rowStrideBytes,
                   const size_t* rowOf, size_t count, size_t dim,
                   int32_t* out) {
  dotMultiVnni_<false>(
      qx,
      [&](size_t j) {
        return reinterpret_cast<const int8_t*>(base +
                                               rowOf[j] * rowStrideBytes);
      },
      count, dim, out, 0);
}

#else  // !QL_VEC_I8_X86 -- inert stubs (never selected; probe false).

bool vnniAvailable() { return false; }
int32_t dotPair(const int8_t*, const int8_t*, size_t) { return 0; }
void dotBlockVnni(const uint8_t*, const char*, size_t, size_t, size_t,
                  int32_t*) {}
void dotGatherVnni(const uint8_t*, const char*, size_t, const size_t*, size_t,
                   size_t, int32_t*) {}
float angularFinalize(float, float, float) { return 0.f; }

#endif

}  // namespace qlever::vector::i8kernels
