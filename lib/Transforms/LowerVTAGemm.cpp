#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct LowerVTAGemmPass
    : public PassWrapper<LowerVTAGemmPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-vta-gemm"; }
  StringRef getDescription() const final {
    return "Lower single-block vta.gemm to vtaisa ISA ops";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vtaisa::VTAISADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<vta::GemmOp> targets;
    module.walk([&](vta::GemmOp g) { targets.push_back(g); });

    // Fixed buffer logical base addresses (page-aligned by the upstream
    // dram_allocation; identical across 16x16 / 32x32 since a single layer
    // fits in one 4 KiB page). Per-block offsets are derived from these.
    constexpr int64_t kUopBase = 5120; // 0x1400
    constexpr int64_t kInpBase = 64;   // 0x40
    constexpr int64_t kWgtBase = 8;    // 0x8
    constexpr int64_t kAccBase = 192;  // 0xc0
    constexpr int64_t kOutBase = 256;  // 0x100

    for (vta::GemmOp g : targets) {
      auto lhsT = g.lhs().getType().dyn_cast<MemRefType>();
      auto rhsT = g.rhs().getType().dyn_cast<MemRefType>();
      auto accT = g.acc().getType().dyn_cast<MemRefType>();
      if (!lhsT || !rhsT || !accT) {
        g.emitError("lower-vta-gemm: expected memref operands");
        signalPassFailure();
        return;
      }
      // Block dimensions (each matrix dim is a multiple of 16, enforced by the
      // op verifier): Mb x Kb (A), Kb x Nb (B), Mb x Nb (C).
      const int64_t Mb = lhsT.getShape()[0] / 16;
      const int64_t Kb = lhsT.getShape()[1] / 16;
      const int64_t Nb = rhsT.getShape()[1] / 16;
      const int64_t nbA = Mb * Kb, nbB = Kb * Nb, nbC = Mb * Nb;

      // Phase-3 supports the no-overfit (CASE 1) single-step schedule, where
      // all blocks fit on-chip at once. Reject anything that would require
      // multi-step tiling (left to a later increment).
      // Upstream block buffer capacities (default config): INP/ACC/OUT 128,
      // WGT 1024 blocks.
      if (nbA >= 128 || nbB >= 1024 || nbC >= 128) {
        g.emitError("lower-vta-gemm: matrix too large for single-step schedule "
                    "(overfit not yet supported)");
        signalPassFailure();
        return;
      }

      // Block schedule, replicating upstream get_gemm_operations: row-major
      // block index idx = row * cols + col. For each loaded A[i,k] block, pair
      // with every B[k,j] block to accumulate C[i,j]. Emit in (a asc, b asc)
      // order so the resulting UOP table matches the Python compiler byte for
      // byte.
      SmallVector<int64_t> uopDst, uopSrc, uopWgt;
      // UOP 0 is the reset placeholder (used by the reset GEMM's uop[0,1)).
      uopDst.push_back(0);
      uopSrc.push_back(0);
      uopWgt.push_back(0);
      for (int64_t a = 0; a < nbA; ++a) {
        int64_t i = a / Kb, k = a % Kb;
        for (int64_t bb = 0; bb < nbB; ++bb) {
          int64_t bk = bb / Nb, j = bb % Nb;
          if (bk != k)
            continue;
          int64_t cIdx = i * Nb + j;
          uopDst.push_back(cIdx * 16);
          uopSrc.push_back(a * 16);
          uopWgt.push_back(bb);
        }
      }
      const int64_t G = static_cast<int64_t>(uopDst.size()) - 1; // gemm uops

      OpBuilder b(g);
      Location loc = g.getLoc();

      // uop_table: reset uop + G gemm uops (becomes uop.bin in order).
      b.create<vtaisa::UopTableOp>(loc, b.getI64ArrayAttr(uopDst),
                                   b.getI64ArrayAttr(uopSrc),
                                   b.getI64ArrayAttr(uopWgt));

      // reset sequence: LOAD UOP[0] then reset GEMM (primes LD<->CMP pipeline).
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                               false, /*sram_base=*/0, /*dram_base=*/kUopBase,
                               /*y_size=*/1, /*x_size=*/1, /*x_stride=*/1);
      b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                   /*push_prev=*/true, /*push_next=*/false,
                                   /*reset=*/true, /*uop_bgn=*/0, /*uop_end=*/1,
                                   /*loop_out=*/1, /*loop_in=*/16,
                                   /*dst_factor_out=*/16, /*dst_factor_in=*/1);

      // Merged data loads: y_size = number of blocks (block logical addresses
      // are equidistant). INP/ACC stride 16 (16 vectors/block), WGT stride 1.
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                               /*pop_next=*/true, false, false, /*sram_base=*/0,
                               /*dram_base=*/kInpBase, /*y_size=*/nbA,
                               /*x_size=*/16, /*x_stride=*/16);
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT, false, false, false,
                               /*push_next=*/true, /*sram_base=*/0,
                               /*dram_base=*/kWgtBase, /*y_size=*/nbB,
                               /*x_size=*/1, /*x_stride=*/1);
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, /*pop_prev=*/true,
                               false, false, false, /*sram_base=*/0,
                               /*dram_base=*/kAccBase, /*y_size=*/nbC,
                               /*x_size=*/16, /*x_stride=*/16);

      // LOAD the G gemm uops (uop.bin index 1..G -> dram = base + 1).
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                               false, /*sram_base=*/0,
                               /*dram_base=*/kUopBase + 1, /*y_size=*/1,
                               /*x_size=*/G, /*x_stride=*/G);
      // Main GEMM over the G uops.
      b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                   /*push_prev=*/true, /*push_next=*/true,
                                   /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/G,
                                   /*loop_out=*/1, /*loop_in=*/16,
                                   /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                   /*src_factor_out=*/0, /*src_factor_in=*/1);

      // One STORE per output block. Dependency bits follow the CASE-1
      // single-step semaphore pattern: first STORE pops the compute-ready
      // token, last STORE pushes the store-done token (for nbC==1 the single
      // STORE carries both, matching the 16x16 golden).
      for (int64_t blk = 0; blk < nbC; ++blk) {
        bool popPrev = (blk == 0);
        bool pushPrev = (blk == nbC - 1);
        b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT, popPrev, false,
                                  pushPrev, false, /*sram_base=*/blk * 16,
                                  /*dram_base=*/kOutBase + blk * 16,
                                  /*y_size=*/1, /*x_size=*/16, /*x_stride=*/16);
      }

      // Termination NOPs (drain the LD<->CMP / ST<->CMP tokens) + FINISH.
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/true, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, /*pop_prev=*/true,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);
      b.create<vtaisa::FinishOp>(loc);

      g.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createLowerVTAGemmPass() {
  return std::make_unique<LowerVTAGemmPass>();
}

void mlir::vta::registerVTAPasses() {
  PassRegistration<LowerVTAGemmPass>();
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createConvertLinalgToVTAPass();
  });
}
