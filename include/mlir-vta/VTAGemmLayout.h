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

/// Compute the layout from block dimensions Mb=M/16, Kb=K/16, Nb=N/16, starting
/// the page-aligned cursor at `baseCursor`. For multi-layer compilation each
/// layer passes the previous layer's `lastPhys` as `baseCursor`, so physical
/// addresses (and thus logical = phys/divisor) climb monotonically across
/// layers, exactly as the upstream dram_allocation threads its base_addr.
inline GemmLayout computeGemmLayoutAt(int64_t Mb, int64_t Kb, int64_t Nb,
                                      int64_t baseCursor) {
  constexpr int64_t kPage = 0x1000;   // 4 KiB
  constexpr int64_t kBlockBytes = 1024; // 16x16 int32
  const int64_t nbA = Mb * Kb, nbB = Kb * Nb, nbC = Mb * Nb;
  const int64_t G = Mb * Kb * Nb; // number of gemm micro-ops

  auto nextPage = [](int64_t cur) { return (cur / kPage + 1) * kPage; };
  int64_t cur = baseCursor;
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

/// Single-layer convenience: cursor starts at 0 (matches a layer compiled on
/// its own, and the 16/32/rectangular single-layer goldens).
inline GemmLayout computeGemmLayout(int64_t Mb, int64_t Kb, int64_t Nb) {
  return computeGemmLayoutAt(Mb, Kb, Nb, /*baseCursor=*/0);
}

// ---------------------------------------------------------------------------
// Page-aligned DRAM allocation for a pure ALU layer (no INP/WGT).
// Replicates the upstream dram_allocation for an ALU-only operation:
//   object order: ACC / OUT / UOP / INSN (no INP, no WGT)
// Matches alu_16x16 golden: ACC=0x40 OUT=0x80 UOP=0xc00 (all logical).
// ---------------------------------------------------------------------------
struct AluLayout {
  int64_t accPhys, outPhys, uopPhys, insnPhys;
  int64_t accLogical, outLogical, uopLogical, insnLogical;
  int64_t numInsns;
  int64_t lastPhys;
};

/// Compute ALU layout for `nbVec` vectors (rows), `G` alu micro-ops,
/// starting page-aligned cursor at `baseCursor`.
/// numInsns = 6 + nbVec  (LOAD UOP reset + LOAD ACC + LOAD UOP alu +
///                         ALU insn + nbVec×STORE + NOP-LOAD + NOP-UOP + FINISH)
///          = 6 + nbVec
inline AluLayout computeAluLayoutAt(int64_t nbVec, int64_t G,
                                     int64_t baseCursor) {
  constexpr int64_t kPage = 0x1000;
  constexpr int64_t kBlockBytes = 1024; // 16×16 int32

  auto nextPage = [](int64_t cur) { return (cur / kPage + 1) * kPage; };
  int64_t cur = baseCursor;
  auto alloc = [&](int64_t sizeBytes, int64_t divisor, int64_t &phys,
                   int64_t &logical) {
    phys = nextPage(cur);
    logical = phys / divisor;
    cur = phys + sizeBytes - 1;
  };

  // nbVec vectors of 16×int32 = nbVec × 1024 bytes (1 block = 16×16 elements)
  // When nbVec is a multiple of 16 rows (one standard 16×16 block = 1 block).
  // General: ceil(nbVec/16) blocks × 1024 bytes? No: each *row* is 16 elements
  // = 64 bytes. But upstream treats ACC as blocks of 16 rows each.
  // For 16×16: 1 block = 1024 bytes; for M×16: ceil(M/16) blocks.
  const int64_t nbBlocks = (nbVec + 15) / 16;

  AluLayout L;
  alloc(nbBlocks * kBlockBytes, /*divisor=*/64, L.accPhys, L.accLogical);
  alloc(nbBlocks * kBlockBytes, /*divisor=*/64, L.outPhys, L.outLogical);
  // UOP: 1 reset uop + G alu uops = (1+G) entries × 4 bytes
  alloc((1 + G) * 4, /*divisor=*/4, L.uopPhys, L.uopLogical);
  // numInsns: LOAD UOP(reset) + ALU reset + LOAD ACC + LOAD UOP(alu) + ALU main
  //           + nbVec × STORE + NOP-LOAD + NOP-UOP + FINISH
  //         = 5 + nbVec + 3 = 8 + nbVec
  L.numInsns = 8 + nbVec;
  alloc(L.numInsns * 16, /*divisor=*/16, L.insnPhys, L.insnLogical);
  L.lastPhys = L.insnPhys + L.numInsns * 16 - 1;
  return L;
}

inline AluLayout computeAluLayout(int64_t nbVec, int64_t G) {
  return computeAluLayoutAt(nbVec, G, /*baseCursor=*/0);
}

} // namespace vta
} // namespace mlir
