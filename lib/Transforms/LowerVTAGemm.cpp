#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
#include "mlir-vta/VTAGemmLayout.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>

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

    // If vta-dram-allocation ran, per-layer bases are recorded on the module and
    // climb across layers; otherwise each gemm self-allocates from cursor 0.
    auto layersAttr = module->getAttrOfType<ArrayAttr>("vta.layers");

    for (auto it : llvm::enumerate(targets)) {
      vta::GemmOp g = it.value();
      DictionaryAttr layerDict;
      if (layersAttr && it.index() < layersAttr.size())
        layerDict = layersAttr[it.index()].dyn_cast<DictionaryAttr>();
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

      // Upstream SRAM block buffer capacities (default vta_config.json).
      constexpr int64_t kInpBuf = 128, kWgtBuf = 1024, kAccBuf = 128;
      const bool isOverfit = (nbA >= kInpBuf || nbB >= kWgtBuf || nbC >= kAccBuf);

      // Strategy-1 parameters (used for both overfit and CASE-1 paths).
      // CASE-1: delta == Kb, nbDelta == 1, remainder == 0 (single step covers all).
      const int64_t bufSize = std::min({kInpBuf, kWgtBuf, kAccBuf});
      const int64_t delta = std::min(bufSize, Kb);
      const int64_t nbDelta = Kb / delta;
      const int64_t remainder = Kb % delta;

      // Build the UOP table.
      //
      // CASE-1 (no overfit): UOP entries use global block indices (a*16, bb) so
      // that the GEMM instruction's loop factors expand them into per-row
      // addresses.  This matches the Phase-3 golden byte-for-byte.
      //
      // Overfit (Strategy-1): Each step loads delta A and B blocks into SRAM
      // starting at offset 0, so UOP src/wgt use SRAM-local indices (k_local).
      // The UOP entries for all C-block/delta-step/remainder combinations are
      // appended in order; each group is independently loaded by its own
      // "LOAD UOP" instruction.
      SmallVector<int64_t> uopDst, uopSrc, uopWgt;
      // UOP 0 is the reset placeholder (used by the reset GEMM's uop[0,1)).
      uopDst.push_back(0);
      uopSrc.push_back(0);
      uopWgt.push_back(0);

      if (!isOverfit) {
        // CASE-1: global-index UOP entries (Phase-3 existing logic).
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
      } else {
        // Strategy-1: SRAM-local-index UOP entries.
        // For each C block, for each delta-step, for each k_local in [0,delta):
        //   dst = 0 (C block always at SRAM offset 0)
        //   src = k_local * 16 (A block at SRAM offset k_local)
        //   wgt = k_local      (B block at SRAM offset k_local)
        // Followed by remainder entries (k_local in [0,remainder)) if any.
        for (int64_t idx = 0; idx < nbC; ++idx) {
          for (int64_t dStep = 0; dStep < nbDelta; ++dStep) {
            for (int64_t kLocal = 0; kLocal < delta; ++kLocal) {
              uopDst.push_back(0);
              uopSrc.push_back(kLocal * 16);
              uopWgt.push_back(kLocal);
            }
          }
          if (remainder > 0) {
            for (int64_t kLocal = 0; kLocal < remainder; ++kLocal) {
              uopDst.push_back(0);
              uopSrc.push_back(kLocal * 16);
              uopWgt.push_back(kLocal);
            }
          }
        }
      }
      const int64_t G = static_cast<int64_t>(uopDst.size()) - 1; // gemm uops

      // Page-aligned DRAM layout (replicates upstream dram_allocation). Bases
      // shift with matrix size and, for multi-layer, climb across layers.
      int64_t kUopBase, kInpBase, kWgtBase, kAccBase, kOutBase;
      if (layerDict) {
        auto getI = [&](StringRef key) {
          return layerDict.getAs<IntegerAttr>(key).getInt();
        };
        kUopBase = getI("uop_log");
        kInpBase = getI("inp_log");
        kWgtBase = getI("wgt_log");
        kAccBase = getI("acc_log");
        kOutBase = getI("out_log");
      } else {
        const vta::GemmLayout L = vta::computeGemmLayout(Mb, Kb, Nb);
        kUopBase = L.uopLogical;
        kInpBase = L.inpLogical;
        kWgtBase = L.wgtLogical;
        kAccBase = L.accLogical;
        kOutBase = L.outLogical;
      }

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

      if (!isOverfit) {
        // CASE-1: single-step schedule. All blocks fit on-chip simultaneously.
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
      } else {
        // Strategy-1: multi-step overfit schedule.
        //
        // For each C block (idx in [0, nbC)):
        //   For each delta-step (dStep in [0, nbDelta)):
        //     LOAD INP: delta blocks from DRAM, pop_next=1 (consume CMP->LD)
        //     LOAD WGT: delta blocks from DRAM, push_next=1 (produce LD->CMP)
        //     LOAD ACC: only for the first C block's first step, pop_prev=1 (consume LD->CMP)
        //     LOAD UOP: delta gemm uops (no dep on delta-steps; pop_prev on remainder)
        //     GEMM: push_prev=1 (produce CMP->LD for next step's LOAD INP);
        //           push_next=1 only on the very last step of the last C block (has STORE)
        //   If remainder > 0:
        //     LOAD INP: remainder blocks, pop_next=1
        //     LOAD WGT: remainder blocks, push_next=1
        //     LOAD UOP: remainder uops, pop_prev=1 (consume LD->CMP — no ACC LOAD here)
        //     GEMM: push_prev=1; push_next=1 if last C block
        //   STORE OUT: for current C block, pop_prev=1 (consume CMP->ST), push_prev=1 (produce ST->CMP)
        //
        // UOP cursor tracks the position of the next group of uops within the
        // UOP DRAM region (relative to UOP_base + 1).
        int64_t uopCursor = 0;

        for (int64_t idx = 0; idx < nbC; ++idx) {
          int64_t ci = idx / Nb; // row index in C block grid
          int64_t cj = idx % Nb; // col index in C block grid
          bool lastC = (idx == nbC - 1);

          for (int64_t dStep = 0; dStep < nbDelta; ++dStep) {
            bool isLastStepForC = (dStep == nbDelta - 1 && remainder == 0);
            bool isAbsoluteLastStep = isLastStepForC && lastC;
            // Global DRAM block indices for this delta-step's A and B tiles.
            int64_t aStart = ci * Kb + dStep * delta;
            int64_t bStart = dStep * delta * Nb + cj;

            // LOAD INP: delta blocks, starting at aStart*16 logical offset.
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
                                     false, /*pop_next=*/true, false, false,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kInpBase + aStart * 16,
                                     /*y_size=*/delta,
                                     /*x_size=*/16, /*x_stride=*/16);

            // LOAD WGT: delta blocks, starting at bStart logical offset.
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT,
                                     false, false, false, /*push_next=*/true,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kWgtBase + bStart,
                                     /*y_size=*/delta,
                                     /*x_size=*/1, /*x_stride=*/1);

            // LOAD ACC: only on the first step of each C block.
            // pop_prev consumes the LD->CMP token produced by LOAD WGT above
            // (or the CMP->LD / ST->CMP token, depending on step context).
            // Semaphore analysis:
            //   - First C block, first step: consume LD->CMP from LOAD WGT.
            //   - Subsequent C blocks, first step: consume ST->CMP from previous STORE.
            // In both cases pop_prev=1 is correct (upstream uses same bit).
            if (dStep == 0) {
              b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC,
                                       /*pop_prev=*/true, false, false, false,
                                       /*sram_base=*/0,
                                       /*dram_base=*/kAccBase + idx * 16,
                                       /*y_size=*/1,
                                       /*x_size=*/16, /*x_stride=*/16);
            }

            // LOAD UOP: delta gemm uop entries (no dep bits on delta-steps).
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
                                     false, false, false, false,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kUopBase + 1 + uopCursor,
                                     /*y_size=*/1,
                                     /*x_size=*/delta, /*x_stride=*/delta);

            // GEMM: push_prev=1 always (produce CMP->LD for next LOAD INP).
            //       push_next=1 only on the absolute last step (signals STORE).
            b.create<vtaisa::GemmInsnOp>(loc, false, false,
                                         /*push_prev=*/true,
                                         /*push_next=*/isAbsoluteLastStep,
                                         /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/delta,
                                         /*loop_out=*/1, /*loop_in=*/16,
                                         /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                         /*src_factor_out=*/0, /*src_factor_in=*/1);

            uopCursor += delta;
          }

          if (remainder > 0) {
            bool isAbsoluteLastStep = lastC;
            int64_t aStart = ci * Kb + nbDelta * delta;
            int64_t bStart = nbDelta * delta * Nb + cj;

            // LOAD INP: remainder blocks.
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
                                     false, /*pop_next=*/true, false, false,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kInpBase + aStart * 16,
                                     /*y_size=*/remainder,
                                     /*x_size=*/16, /*x_stride=*/16);

            // LOAD WGT: remainder blocks.
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT,
                                     false, false, false, /*push_next=*/true,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kWgtBase + bStart,
                                     /*y_size=*/remainder,
                                     /*x_size=*/1, /*x_stride=*/1);

            // LOAD UOP: pop_prev=1 to consume the LD->CMP token produced by
            // LOAD WGT above (no LOAD ACC in remainder step).
            b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
                                     /*pop_prev=*/true, false, false, false,
                                     /*sram_base=*/0,
                                     /*dram_base=*/kUopBase + 1 + uopCursor,
                                     /*y_size=*/1,
                                     /*x_size=*/remainder, /*x_stride=*/remainder);

            // GEMM: push_prev=1 (CMP->LD), push_next=1 on absolute last step.
            b.create<vtaisa::GemmInsnOp>(loc, false, false,
                                         /*push_prev=*/true,
                                         /*push_next=*/isAbsoluteLastStep,
                                         /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/remainder,
                                         /*loop_out=*/1, /*loop_in=*/16,
                                         /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                         /*src_factor_out=*/0, /*src_factor_in=*/1);

            uopCursor += remainder;
          }

          // STORE OUT: after all steps for this C block are done.
          // pop_prev=1 consumes the CMP->ST token from the last GEMM's push_next.
          // push_prev=1 produces the ST->CMP token for the next C block's LOAD ACC.
          b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
                                    /*pop_prev=*/true, false,
                                    /*push_prev=*/true, false,
                                    /*sram_base=*/0,
                                    /*dram_base=*/kOutBase + idx * 16,
                                    /*y_size=*/1, /*x_size=*/16, /*x_stride=*/16);
        }
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
  ::mlir::registerPass([]() -> std::unique_ptr<Pass> {
    return mlir::vta::createVTADramAllocationPass();
  });
}
