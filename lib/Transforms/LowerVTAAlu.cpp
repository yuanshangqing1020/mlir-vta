// LowerVTAAlu: lower vta.alu to vtaisa.* ISA instructions.
//
// Instruction sequence (matching the upstream standalone-vta alu_strategy,
// ADD_IMM variant, verified byte-by-byte against golden alu_add_imm_16x16):
//
//   LOAD UOP(reset)    dram=uopBase,   y=1, x=1,  stride=1
//   GEMM reset         push_prev=1,    uop[0,1), lp_out=1, lp_in=16
//   LOAD ACC           dram=accBase,   y=nbBlocks, x=16, stride=16
//   LOAD UOP(alu)      dram=uopBase+1, y=1, x=G, stride=G
//   ALU main           push_next=1(if store follows), uop[0,G),
//                      lp_out=1, lp_in=1, use_imm, imm, alu_opcode
//   STORE OUT × nbVec  sram=i, dram=outBase+i, y=1, x=1, stride=1
//   NOP-LOAD(INP)      y=0, x=0, dram=0, pop_next=1, push_next=1
//   NOP-LOAD(UOP)      y=0, x=0, dram=0, pop_prev=1, pop_next=1
//   FINISH
//
// UOP table (1 reset + G alu uops):
//   uop[0]   = (0, 0, 0)            reset placeholder
//   uop[1+i] = (i, 0, 0)  i∈[0,G)  dst=row, src=0, wgt=0
//
// where G = nbVec (one uop per output row/vector), nbBlocks = ceil(nbVec/16).
//
// Dep bits (--vta-semaphore-derive produces the same result, but we emit
// them inline here for correctness without requiring the extra pass):
//   GEMM reset : push_prev=1
//   ALU main   : push_next=1  (STORE follows)
//   STORE[0]   : pop_prev=1
//   STORE[last]: push_prev=1
//   NOP-LOAD(INP): pop_next=1, push_next=1
//   NOP-LOAD(UOP): pop_prev=1, pop_next=1

#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
#include "mlir-vta/VTAGemmLayout.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct LowerVTAAluPass
    : public PassWrapper<LowerVTAAluPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-vta-alu"; }
  StringRef getDescription() const final {
    return "Lower vta.alu to vtaisa ISA instructions (ADD_IMM / MAX / etc.)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vtaisa::VTAISADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<vta::AluOp> targets;
    module.walk([&](vta::AluOp a) { targets.push_back(a); });

    for (auto &op : targets) {
      auto accT = op.acc().getType().dyn_cast<MemRefType>();
      if (!accT) {
        op.emitError("lower-vta-alu: expected memref acc operand");
        signalPassFailure();
        return;
      }

      // Shape: [M, 16] where M = number of 16-element row vectors.
      const int64_t nbVec   = accT.getShape()[0]; // M rows
      const int64_t G       = nbVec;              // one ALU uop per row
      const int64_t nbBlocks = (nbVec + 15) / 16; // ceil(M/16) 16×16 blocks

      // Compute DRAM layout.
      auto L = vta::computeAluLayout(nbVec, G);

      OpBuilder b(op);
      Location loc = op.getLoc();

      // ---------------------------------------------------------------
      // UOP table: 1 reset + G alu uops.
      // ---------------------------------------------------------------
      SmallVector<int64_t> uopDst, uopSrc, uopWgt;
      uopDst.push_back(0); uopSrc.push_back(0); uopWgt.push_back(0); // reset
      for (int64_t i = 0; i < G; ++i) {
        uopDst.push_back(i);   // dst = row index in SRAM
        uopSrc.push_back(0);   // src = 0 (not used for imm ALU)
        uopWgt.push_back(0);   // wgt = 0
      }
      b.create<vtaisa::UopTableOp>(loc,
          b.getI64ArrayAttr(uopDst),
          b.getI64ArrayAttr(uopSrc),
          b.getI64ArrayAttr(uopWgt));

      // ---------------------------------------------------------------
      // LOAD UOP reset (1 uop from dram=uopBase, SRAM[0]=reset uop).
      // ---------------------------------------------------------------
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
          /*pop_prev=*/false, /*pop_next=*/false,
          /*push_prev=*/false, /*push_next=*/false,
          /*sram_base=*/0, /*dram_base=*/L.uopLogical,
          /*y_size=*/1, /*x_size=*/1, /*x_stride=*/1);

      // ---------------------------------------------------------------
      // GEMM reset (opcode=2 with reset=true, push_prev=1).
      // This primes the LD<->CMP pipeline, same as in LowerVTAGemm.
      // lp_in=16, dst_factor_out=16, dst_factor_in=1 (same pattern).
      // ---------------------------------------------------------------
      b.create<vtaisa::GemmInsnOp>(loc,
          /*pop_prev=*/false, /*pop_next=*/false,
          /*push_prev=*/true, /*push_next=*/false,
          /*reset=*/true,
          /*uop_bgn=*/0, /*uop_end=*/1,
          /*loop_out=*/1, /*loop_in=*/16,
          /*dst_factor_out=*/16, /*dst_factor_in=*/1,
          /*src_factor_out=*/0,  /*src_factor_in=*/0);

      // ---------------------------------------------------------------
      // LOAD ACC: all nbBlocks blocks from DRAM (one logical load).
      // y_size=nbBlocks (number of 16×16 blocks), x_size=16, stride=16.
      // ---------------------------------------------------------------
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC,
          /*pop_prev=*/false, /*pop_next=*/false,
          /*push_prev=*/false, /*push_next=*/false,
          /*sram_base=*/0, /*dram_base=*/L.accLogical,
          /*y_size=*/nbBlocks, /*x_size=*/16, /*x_stride=*/16);

      // ---------------------------------------------------------------
      // LOAD UOP alu: G uops from dram=uopBase+1, SRAM[0..G-1].
      // ---------------------------------------------------------------
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
          /*pop_prev=*/false, /*pop_next=*/false,
          /*push_prev=*/false, /*push_next=*/false,
          /*sram_base=*/0, /*dram_base=*/L.uopLogical + 1,
          /*y_size=*/1, /*x_size=*/G, /*x_stride=*/G);

      // ---------------------------------------------------------------
      // ALU main: push_next=1 (STORE follows), uop[0,G).
      // loop_out=1, loop_in=1 (each uop processes 16 elements).
      // ---------------------------------------------------------------
      b.create<vtaisa::AluInsnOp>(loc,
          /*pop_prev=*/false, /*pop_next=*/false,
          /*push_prev=*/false, /*push_next=*/true,
          /*reset=*/false,
          /*uop_bgn=*/0, /*uop_end=*/G,
          /*loop_out=*/1, /*loop_in=*/1,
          /*dst_factor_out=*/0, /*dst_factor_in=*/0,
          /*src_factor_out=*/0, /*src_factor_in=*/0,
          /*alu_opcode=*/op.alu_opcode(),
          /*use_imm=*/op.use_imm(),
          /*imm=*/op.imm());

      // ---------------------------------------------------------------
      // STORE OUT: one STORE per vector row.
      // First STORE: pop_prev=1; last STORE: push_prev=1.
      // ---------------------------------------------------------------
      for (int64_t i = 0; i < nbVec; ++i) {
        bool isFirst = (i == 0);
        bool isLast  = (i == nbVec - 1);
        b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT,
            /*pop_prev=*/isFirst, /*pop_next=*/false,
            /*push_prev=*/isLast, /*push_next=*/false,
            /*sram_base=*/i, /*dram_base=*/L.outLogical + i,
            /*y_size=*/1, /*x_size=*/1, /*x_stride=*/1);
      }

      // ---------------------------------------------------------------
      // Termination NOPs (same as LowerVTAGemm).
      // NOP-LOAD(INP): pop_next=1, push_next=1, y=0, x=0.
      // NOP-LOAD(UOP): pop_prev=1, pop_next=1, y=0, x=0.
      // ---------------------------------------------------------------
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP,
          /*pop_prev=*/false, /*pop_next=*/true,
          /*push_prev=*/false, /*push_next=*/true,
          /*sram_base=*/0, /*dram_base=*/0,
          /*y_size=*/0, /*x_size=*/0, /*x_stride=*/0);

      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP,
          /*pop_prev=*/true, /*pop_next=*/true,
          /*push_prev=*/false, /*push_next=*/false,
          /*sram_base=*/0, /*dram_base=*/0,
          /*y_size=*/0, /*x_size=*/0, /*x_stride=*/0);

      // FINISH
      b.create<vtaisa::FinishOp>(loc);

      // Remove the original vta.alu op.
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> mlir::vta::createLowerVTAAluPass() {
  return std::make_unique<LowerVTAAluPass>();
}
