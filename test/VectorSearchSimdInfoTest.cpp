// Copyright 2026 The QLever Authors, in particular:
//
// 2026 Artem <artem@rem.sh>

// You may not use this file except in compliance with the Apache 2.0 License,
// which can be found in the `LICENSE` file at the root of the QLever project.

// Tests for the pure NumKong capability-bitmask -> ISA-name mapping
// (`numkongActiveIsaString`) behind the one-time "Vector search SIMD"
// startup log of the vector-index load hook.

#include <gtest/gtest.h>

#include <string>

#include "services/vectorSearch/VectorIndexExtension.h"

#if defined(USEARCH_USE_NUMKONG) && USEARCH_USE_NUMKONG
#include <numkong/numkong.h>
#endif

using qlever::vector::numkongActiveIsaString;

#if defined(USEARCH_USE_NUMKONG) && USEARCH_USE_NUMKONG

// _____________________________________________________________________________
TEST(VectorSearchSimdInfo, isaStringNamesExactlyTheSetBits) {
  std::string isa =
      numkongActiveIsaString(nk_cap_skylake_k | nk_cap_sapphire_k);
  EXPECT_NE(isa.find("skylake"), std::string::npos) << isa;
  EXPECT_NE(isa.find("sapphire"), std::string::npos) << isa;
  // Bits that are NOT set must not be reported, and a mask with SIMD bits is
  // never the scalar fallback.
  EXPECT_EQ(isa.find("haswell"), std::string::npos) << isa;
  EXPECT_EQ(isa.find("serial"), std::string::npos) << isa;
}

// _____________________________________________________________________________
TEST(VectorSearchSimdInfo, isaStringCoversTheX86Generations) {
  std::string all = numkongActiveIsaString(
      nk_cap_haswell_k | nk_cap_skylake_k | nk_cap_icelake_k | nk_cap_genoa_k |
      nk_cap_sapphire_k | nk_cap_sapphireamx_k | nk_cap_turin_k);
  for (const char* name :
       {"haswell/avx2", "skylake/avx512", "icelake/avx512-vnni",
        "genoa/avx512-bf16", "sapphire/avx512-fp16", "amx", "turin"}) {
    EXPECT_NE(all.find(name), std::string::npos) << all << " misses " << name;
  }
}

// _____________________________________________________________________________
TEST(VectorSearchSimdInfo, isaStringCoversArm) {
  std::string arm = numkongActiveIsaString(nk_cap_neon_k | nk_cap_sve_k);
  EXPECT_NE(arm.find("neon"), std::string::npos) << arm;
  EXPECT_NE(arm.find("sve"), std::string::npos) << arm;
}

// _____________________________________________________________________________
TEST(VectorSearchSimdInfo, serialOnlyReportsTheScalarFallback) {
  EXPECT_EQ(numkongActiveIsaString(nk_cap_serial_k), "serial(scalar)");
  EXPECT_EQ(numkongActiveIsaString(0), "serial(scalar)");
}

#else

// _____________________________________________________________________________
TEST(VectorSearchSimdInfo, withoutNumkongEverythingIsScalar) {
  // Built without NumKong: whatever the mask, there is no SIMD kernel.
  EXPECT_EQ(numkongActiveIsaString(0), "serial(scalar)");
  EXPECT_EQ(numkongActiveIsaString(~uint64_t{0}), "serial(scalar)");
}

#endif
