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

    for (vta::GemmOp g : targets) {
      // Phase 2: still only a single 16x16x16 block; derive dims from operands.
      auto is16 = [](Value v) {
        auto mt = v.getType().dyn_cast<MemRefType>();
        return mt && mt.getRank() == 2 && mt.getShape()[0] == 16 &&
               mt.getShape()[1] == 16;
      };
      if (!is16(g.lhs()) || !is16(g.rhs()) || !is16(g.acc())) {
        g.emitError("lower-vta-gemm: only 16x16 memref operands supported");
        signalPassFailure();
        return;
      }

      OpBuilder b(g);
      Location loc = g.getLoc();

      // 1. uop_table dst=[0,0] src=[0,0] wgt=[0,0]
      b.create<vtaisa::UopTableOp>(loc, b.getI64ArrayAttr({0, 0}),
                                b.getI64ArrayAttr({0, 0}),
                                b.getI64ArrayAttr({0, 0}));

      // 2. load UOP sram=0 dram=5120 y=1 x=1 stride=1
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, /*pop_prev=*/false,
                               /*pop_next=*/false, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/5120, /*y_size=*/1, /*x_size=*/1,
                               /*x_stride=*/1);

      // 3. gemm_insn push_prev reset uop[0,1) loop_out=1 loop_in=16
      //    dst_factor_out=16 dst_factor_in=1
      b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                   /*push_prev=*/true, /*push_next=*/false,
                                   /*reset=*/true, /*uop_bgn=*/0, /*uop_end=*/1,
                                   /*loop_out=*/1, /*loop_in=*/16,
                                   /*dst_factor_out=*/16, /*dst_factor_in=*/1);

      // 4. load INP pop_next sram=0 dram=64 y=1 x=16 stride=16
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/64, /*y_size=*/1, /*x_size=*/16,
                               /*x_stride=*/16);

      // 5. load WGT push_next sram=0 dram=8 y=1 x=1 stride=1
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::WGT, /*pop_prev=*/false,
                               /*pop_next=*/false, /*push_prev=*/false,
                               /*push_next=*/true, /*sram_base=*/0,
                               /*dram_base=*/8, /*y_size=*/1, /*x_size=*/1,
                               /*x_stride=*/1);

      // 6. load ACC pop_prev sram=0 dram=192 y=1 x=16 stride=16
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, /*pop_prev=*/true,
                               /*pop_next=*/false, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/192, /*y_size=*/1, /*x_size=*/16,
                               /*x_stride=*/16);

      // 7. load UOP sram=0 dram=5121 y=1 x=1 stride=1
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, /*pop_prev=*/false,
                               /*pop_next=*/false, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/5121, /*y_size=*/1, /*x_size=*/1,
                               /*x_stride=*/1);

      // 8. gemm_insn push_prev push_next uop[0,1) loop_out=1 loop_in=16
      //    dst_factor_in=1 src_factor_in=1
      b.create<vtaisa::GemmInsnOp>(loc, /*pop_prev=*/false, /*pop_next=*/false,
                                   /*push_prev=*/true, /*push_next=*/true,
                                   /*reset=*/false, /*uop_bgn=*/0, /*uop_end=*/1,
                                   /*loop_out=*/1, /*loop_in=*/16,
                                   /*dst_factor_out=*/0, /*dst_factor_in=*/1,
                                   /*src_factor_out=*/0, /*src_factor_in=*/1);

      // 9. store OUT pop_prev push_prev sram=0 dram=256 y=1 x=16 stride=16
      b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT, /*pop_prev=*/true,
                                /*pop_next=*/false, /*push_prev=*/true,
                                /*push_next=*/false, /*sram_base=*/0,
                                /*dram_base=*/256, /*y_size=*/1, /*x_size=*/16,
                                /*x_stride=*/16);

      // 10. load INP pop_next push_next sram=0 dram=0 y=0 x=0 stride=0
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, /*pop_prev=*/false,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/true, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);

      // 11. load UOP pop_prev pop_next sram=0 dram=0 y=0 x=0 stride=0
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, /*pop_prev=*/true,
                               /*pop_next=*/true, /*push_prev=*/false,
                               /*push_next=*/false, /*sram_base=*/0,
                               /*dram_base=*/0, /*y_size=*/0, /*x_size=*/0,
                               /*x_stride=*/0);

      // 12. finish
      b.create<vtaisa::FinishOp>(loc);

      g.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createLowerVTAGemmPass() {
  return std::make_unique<LowerVTAGemmPass>();
}

void mlir::vta::registerVTAPasses() { PassRegistration<LowerVTAGemmPass>(); }
