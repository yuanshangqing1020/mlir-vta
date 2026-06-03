#pragma once
#include <cstdint>

namespace mlir {
namespace vta {

/// Page-aligned DRAM allocation for a single-layer pure GEMM
/// `C[MxN] = A[MxK] * B[KxN]`, replicating the upstream standalone-vta
/// dram_allocation (4 KiB pages, object order INP/WGT/ACC/OUT/UOP/INSN).
///
/// All addresses are in the units the VTA instructions / memory_addresses.csv
/// expect: `*Logical` are the per-buffer logical base addresses (used as
/// `dram_base` of the LOAD/STORE instructions), `*Phys` are the physical byte
/// addresses (used by memory_addresses.csv). Per-block offsets are derived by
/// the caller (INP/ACC/OUT +16 per block, WGT +1 per block).
struct GemmLayout {
  int64_t inpPhys, wgtPhys, accPhys, outPhys, uopPhys, insnPhys;
  int64_t inpLogical, wgtLogical, accLogical, outLogical, uopLogical,
      insnLogical;
  int64_t numInsns; ///< instruction count = 10 + nbC (CASE-1 single step)
  int64_t lastPhys; ///< last physical DRAM byte allocated (for layers_name.csv)
};

/// Compute the layout from block dimensions Mb=M/16, Kb=K/16, Nb=N/16.
inline GemmLayout computeGemmLayout(int64_t Mb, int64_t Kb, int64_t Nb) {
  constexpr int64_t kPage = 0x1000;   // 4 KiB
  constexpr int64_t kBlockBytes = 1024; // 16x16 int32
  const int64_t nbA = Mb * Kb, nbB = Kb * Nb, nbC = Mb * Nb;
  const int64_t G = Mb * Kb * Nb; // number of gemm micro-ops

  auto nextPage = [](int64_t cur) { return (cur / kPage + 1) * kPage; };
  int64_t cur = 0;
  auto alloc = [&](int64_t sizeBytes, int64_t divisor, int64_t &phys,
                   int64_t &logical) {
    phys = nextPage(cur);
    logical = phys / divisor;
    cur = phys + sizeBytes - 1;
  };

  GemmLayout L;
  alloc(nbA * kBlockBytes, /*divisor=*/64, L.inpPhys, L.inpLogical);
  alloc(nbB * kBlockBytes, /*divisor=*/1024, L.wgtPhys, L.wgtLogical);
  alloc(nbC * kBlockBytes, /*divisor=*/64, L.accPhys, L.accLogical);
  alloc(nbC * kBlockBytes, /*divisor=*/64, L.outPhys, L.outLogical);
  // UOP buffer holds 1 reset uop + G gemm uops, 4 bytes each.
  alloc((1 + G) * 4, /*divisor=*/4, L.uopPhys, L.uopLogical);
  L.numInsns = 10 + nbC;
  alloc(L.numInsns * 16, /*divisor=*/16, L.insnPhys, L.insnLogical);
  L.lastPhys = L.insnPhys + L.numInsns * 16 - 1;
  return L;
}

} // namespace vta
} // namespace mlir
