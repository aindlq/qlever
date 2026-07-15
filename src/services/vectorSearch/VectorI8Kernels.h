// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORI8KERNELS_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORI8KERNELS_H

#include <cstddef>
#include <cstdint>

// Hand-rolled i8 (int8) cosine kernels for the exact COARSE-layer sweep of a
// two-layer index (i8 scan + bf16 rerank) and the single-layer i8 exact scan
// -- the i8 analog of `VectorBf16Kernels`:
//
//  * dotBlockVnni  -- a multi-row AVX-512-VNNI (`vpdpbusd`) dot of one query
//                     against a CONTIGUOUS block of rows: 8 independent
//                     accumulator chains, query reloaded from L1 per 64 B.
//                     `vpdpbusd` MACs 64 u8*i8 products per instruction --
//                     one instruction per row cache line -- so a whole-index
//                     sweep rides the memory-bandwidth wall (the punned
//                     per-row metric is ~7x more core work and caps well
//                     below it; measured 163 vs 23 GB/s of row bytes per
//                     core on an EMR Xeon).
//  * dotGatherVnni -- the same 8-row engine for SCATTERED rows (candidate
//                     gathers), row pointers through a rowOf[] indirection.
//  * dotPair       -- one exact signed i8 dot (`vpmaddwd` after i8->i16 sign
//                     extension); the per-row/scattered fallback engine.
//
// SIGN TRICK: `vpdpbusd` multiplies UNSIGNED by signed bytes. The query is
// biased ONCE per search (`qx[i] = q[i] XOR 0x80`, i.e. `q[i] + 128` as u8),
// so the block kernels compute `dotU = sum (q[i]+128) * r[i]` -- an EXACT
// integer for any i8 inputs (max |dotU| < 2^31 up to dim ~ 260k). The caller
// recovers the true dot via the exact integer identity
//     dot = dotU - 128 * sum(r)
// with `sum(r)` from a per-row sidecar built at open (`rowSums`). This is
// exact for the full i8 range INCLUDING -128 (no clamping anywhere).
//
// FINALIZE: `angularFinalize` converts the three exact integer sums
// (dot, |q|^2, |row|^2) to the angular distance with the SAME
// rsqrt+Newton-Raphson refinement NumKong's `nk_angular_normalize_f32_...`
// uses. It is ONE pinned (noinline, explicit-SSE) instance, so every i8
// exact path -- whole-index block sweep, scattered gather, per-row, the
// single pair -- produces bit-identical distances for the same pair. It can
// differ from usearch's punned metric by ~1 ulp on a small fraction of pairs
// (the punned finalize is compiled under a different ISA context, so its FMA
// contraction differs); that tolerance is accepted exactly like the bf16
// fine layer's batched cosine (the HNSW graph only pre-selects, candidates
// are re-scored by these kernels).
//
// The hot functions carry per-function `target` attributes (AVX-512-VNNI),
// so this TU compiles under the baseline `-march` and stays portable -- the
// same trick as `VectorBf16Kernels`. On a non-x86 / non-VNNI machine the
// `vnniAvailable()` probe returns false and callers keep the punned metric.

namespace qlever::vector::i8kernels {

// Runtime CPU probe (cached; safe to call per query): true iff
// AVX-512-VNNI (+f/bw/vl) is present.
bool vnniAvailable();

// Re-read `QLEVER_VECTOR_SEARCH_PREFETCH` (the contiguous-sweep prefetch
// policy: `off`, `t0x<N>`/`t1x<N>`/`t2x<N>` = hint + N-rows-ahead lead,
// default = the measured winner). The policy is parsed once at first use;
// this refresh exists for the prefetch benchmark, which flips the env
// between timed phases inside one process.
void refreshPrefetchConfigFromEnv();

// The exact integer `sum(v)` and `sum(v^2)` of one i8 vector -- the per-row
// sidecar terms (and the query-side `|q|^2`). Portable scalar (exact for the
// full i8 range including -128); runs once per row at open / once per query.
void rowSums(const int8_t* v, size_t dim, int32_t* sum, int32_t* normSq);

// One exact signed dot product of two i8 vectors (i32; `vpmaddwd` based).
// Bit-identical (it is an exact integer) to `dotU - 128*sum(b)` from the
// block kernels. Only valid when `vnniAvailable()`.
int32_t dotPair(const int8_t* a, const int8_t* b, size_t dim);

// Unsigned-biased dots of one query (`qx` = query XOR 0x80, `dim` bytes)
// against a CONTIGUOUS block of `count` rows starting at `rows`,
// `rowStrideBytes` apart: `out[j] = sum (q[i]+128) * row_j[i]` (exact i32).
// Multi-row unrolled; software-prefetches each row stream ~4 rows ahead
// (T2; disable via QLEVER_VECTOR_SEARCH_PREFETCH=0 -- headroom-dependent:
// neutral at the bandwidth wall, a few % below it). Only valid when
// `vnniAvailable()`.
void dotBlockVnni(const uint8_t* qx, const char* rows, size_t rowStrideBytes,
                  size_t count, size_t dim, int32_t* out);

// The same for SCATTERED rows: `out[j]` gets the biased dot of the row at
// `base + rowOf[j] * rowStrideBytes`. No internal prefetch (the caller
// prefetches the resolved rows, mirroring the bf16 gather). Only valid when
// `vnniAvailable()`.
void dotGatherVnni(const uint8_t* qx, const char* base, size_t rowStrideBytes,
                   const size_t* rowOf, size_t count, size_t dim, int32_t* out);

// The shared cosine finalize over the three EXACT integer sums, as f32:
//   a2 == 0 && b2 == 0  ->  0
//   dot == 0            ->  1
//   else                ->  max(0, 1 - dot * rsqrt(a2) * rsqrt(b2))
// with rsqrt = `_mm_rsqrt_ps` + one Newton-Raphson step (NumKong's
// `nk_angular_normalize_f32_haswell_`). ONE pinned instance -- every i8 path
// calls exactly this function, so all agree to the bit. Works on any x86;
// the (never-selected) non-x86 stub returns 0.
float angularFinalize(float dot, float a2, float b2);

// The rsqrt + one-NR-step refinement of `angularFinalize`, alone: bit-equal
// to the inverse square root `angularFinalize` computes internally for `x`
// (same estimate, same refinement sequence, pinned noinline in the same
// baseline TU). For the per-row inverse-norm sidecar built at open and the
// once-per-query inverse query norm of `angularFinalizeBlock`.
float finalizeInvSqrt(float x);

// The VECTORIZED (16 rows/iteration AVX-512) sibling of calling
// `angularFinalize(dotBiased[j] - 128*rowSums[j], qNormSq, rowNormSq[j])`
// per row, with both inverse norms PRECOMPUTED via `finalizeInvSqrt`:
//   out[j] = max(0, 1 - (f32(dot) * qInv) * rowInvNorms[j]),   dot != 0
//   out[j] = 1,                                                 dot == 0
// BIT-IDENTICAL to the scalar finalize for every row (same integer dot, same
// i32->f32 rounding, the identical rsqrt+NR inverse norms, the same multiply
// order with contraction blocked). PRECONDITION: the query norm is nonzero
// (`qInv = finalizeInvSqrt(qNormSq)`, qNormSq > 0) -- a zero-norm query must
// take the scalar per-row fallback (its `a2 == 0 && b2 == 0 -> 0` convention
// is not vectorized). Requires AVX-512F (guaranteed by `vnniAvailable()`).
void angularFinalizeBlock(const int32_t* dotsBiased, const int32_t* rowSums,
                          const float* rowInvNorms, size_t count, float qInv,
                          float* out);

}  // namespace qlever::vector::i8kernels

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORI8KERNELS_H
