// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

#ifndef QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORBF16KERNELS_H
#define QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORBF16KERNELS_H

#include <cstddef>
#include <cstdint>

// Two hand-rolled bf16 dot-product kernels for the exact fine-layer cosine
// sweep, productionizing the `vector-scan-opt` prototypes:
//
//  * SIMD  -- a multi-row AVX-512-BF16 (`vdpbf16ps`) dot, query held in
//             registers; the whole-scan (contiguous) AND rerank (scattered)
//             engine, unified over a row-pointer callback. ~97 GB/s core
//             ceiling, ~1.5x the NumKong per-pair `nk_dot_bf16`.
//  * AMX   -- the fixed width-1 AMX-BF16 tile GEMM: the query's 36 depth
//             tiles are packed ONCE per search, rows are `_tile_loadd`-ed
//             directly at their store stride, two C accumulators per 32-row
//             block, column 0 extracted. For the contiguous full-scan sweep;
//             the scattered rerank falls through to the SIMD kernel.
//
// Both compute a RAW dot product (bf16 x bf16 -> f32); the cosine finalize
// (`1 - dot * invNormA * invNormB`, clamped at 0) is done by the caller so the
// result is bit-identical to the existing batched path's finalize. The kernel
// functions carry per-function `target` attributes (AMX/AVX-512-BF16) so this
// TU stays compilable under the baseline `-march` -- the SAME portability
// trick NumKong's own per-ISA kernels use. `dim` must be a multiple of 32
// (guaranteed for SigLIP-style embeddings; the caller gates on it).
//
// All are no-ops (compiled to `return`s) on a non-x86 / SIMD-off build; the
// `*Available()` probes return false there and callers keep the punned metric.

namespace qlever::vector::bf16kernels {

// Runtime CPU probes (cached; safe to call per query). `simdAvailable()` is
// true iff AVX-512-BF16 (+vl/bw/f) is present; `amxAvailable()` iff AMX-BF16
// (which also implies the SIMD ISA on every shipping part) is present.
bool simdAvailable();
bool amxAvailable();

// Re-read `QLEVER_VECTOR_SEARCH_PREFETCH` (the contiguous-sweep prefetch
// policy: `off`, `t0x<N>`/`t1x<N>`/`t2x<N>` = hint + N-rows-ahead lead,
// default = the measured winner). Parsed once at first use; this refresh
// exists for the prefetch benchmark, which flips the env between timed
// phases inside one process.
void refreshPrefetchConfigFromEnv();

// Request the process's AMX tile-data permission (Linux `arch_prctl`
// `ARCH_REQ_XCOMP_PERM` for `XFEATURE_XTILEDATA`) and VERIFY it was granted
// (`ARCH_GET_XCOMP_PERM`). Returns true iff the AMX tile state is usable, so a
// kernel/container that refuses the grant (old kernel, seccomp, policy) makes
// the caller DOWNGRADE to the SIMD kernel instead of faulting at `ldtilecfg`.
// Idempotent; the permission is process-wide and inherited by all threads. On
// a non-x86 / SIMD-off build this returns false (no AMX). Gated on the SAME
// feature set as `amxAvailable()`, not NumKong's INT8-inclusive capability.
bool ensureAmxPermission();

// ---- SIMD (AVX-512-BF16) ------------------------------------------------

// Dot products of one bf16 `query` (`dim` elements) against a CONTIGUOUS block
// of `count` rows starting at `rows`, `rowStrideBytes` apart. `out[j]` gets the
// raw f32 dot of row j. Multi-row unrolled; query reloaded from L1 per 64 B.
void dotBlockSimd(const uint16_t* query, const char* rows,
                  size_t rowStrideBytes, size_t count, size_t dim, float* out);

// The same for SCATTERED rows: `out[j]` gets the dot of the row at
// `base + rowOf[j] * rowStrideBytes`. `rowOf` is a caller-provided store-row
// index per output. Unified rerank-gather engine.
void dotGatherSimd(const uint16_t* query, const char* base,
                   size_t rowStrideBytes, const size_t* rowOf, size_t count,
                   size_t dim, float* out);

// A single pair (the `vec:distance` slow path): raw dot of two bf16 vectors.
float dotPairSimd(const uint16_t* a, const uint16_t* b, size_t dim);

// ---- AMX (fixed width-1 tile GEMM) --------------------------------------

// Bytes needed to hold the packed query tiles for `dim` (== `dim/32` tiles of
// 1 KB). The caller allocates this once per search and passes it to
// `packQueryAmx` / `dotBlockAmx`.
size_t packedQuerySizeAmx(size_t dim);

// Pack one bf16 `query` (`dim` elements) into the AMX B-tile layout at `dst`
// (>= `packedQuerySizeAmx(dim)` bytes, 64-B aligned). Done ONCE per search.
void packQueryAmx(const uint16_t* query, size_t dim, void* dst);

// Dot products of the packed query (`packedQuery` from `packQueryAmx`) against
// a CONTIGUOUS block of `count` rows at `rows` (stride `rowStrideBytes`).
// `out[j]` gets the raw f32 dot of row j. `query` is the SAME plain bf16 query
// (used only for a `< 32`-row tail, scored exactly by the SIMD one-row dot so
// no output is left to a masked AMX edge tile). Configures/releases the AMX
// tiles internally (per call; cheap relative to a 1024-row block). Requires the
// AMX tile-data permission to have been granted process-wide (QLever does this
// at index open via `nk_configure_thread`).
void dotBlockAmx(const void* packedQuery, const uint16_t* query,
                 const char* rows, size_t rowStrideBytes, size_t count,
                 size_t dim, float* out);

}  // namespace qlever::vector::bf16kernels

#endif  // QLEVER_SRC_SERVICES_VECTORSEARCH_VECTORBF16KERNELS_H
