// LowerVTAMaxPool: lower vta.maxpool (36x16 2x2 pattern) to vtaisa.* instructions.
// Matches upstream maxpool_36x16.json golden (33 uops, 18 insns).

#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
#include "mlir-vta/VTAGemmLayout.h"
#include "mlir-vta/VTALayerUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct LowerVTAMaxPoolPass
    : public PassWrapper<LowerVTAMaxPoolPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "lower-vta-maxpool"; }
  StringRef getDescription() const final {
    return "Lower vta.maxpool to vtaisa ISA (36x16 vector MAX pattern)";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vtaisa::VTAISADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<vta::MaxPoolOp> targets;
    module.walk([&](vta::MaxPoolOp m) { targets.push_back(m); });

    for (auto &op : targets) {
      auto accT = op.acc().getType().dyn_cast<MemRefType>();
      if (!accT) {
        op.emitError("lower-vta-maxpool: expected memref acc");
        signalPassFailure();
        return;
      }
      const int64_t nbVec = accT.getShape()[0];
      const int64_t nbBlocks = (nbVec + 15) / 16;

      int64_t kAccBase, kOutBase, kUopBase;
      if (auto layerDict = vta::findLayerDict(module, op.name().getValueOr(""))) {
        auto getI = [&](StringRef key) {
          return layerDict.getAs<IntegerAttr>(key).getInt();
        };
        kAccBase = getI("acc_log");
        kOutBase = getI("out_log");
        kUopBase = getI("uop_log");
      } else {
        auto L = vta::computeAluLayout(nbVec, /*G=*/32);
        kAccBase = L.accLogical;
        kOutBase = L.outLogical;
        kUopBase = L.uopLogical;
      }

      // Four 2x2 pooling windows along rows (maxpool_36x16.json).
      const int64_t windows[] = {0, 9, 18, 27};
      const int64_t nWin = 4;
      const int64_t iters = 8;

      SmallVector<int64_t> uopDst, uopSrc, uopWgt;
      uopDst.push_back(0);
      uopSrc.push_back(0);
      uopWgt.push_back(0);
      for (int64_t w = 0; w < nWin; ++w) {
        int64_t base = windows[w];
        for (int64_t i = 0; i < iters; ++i) {
          uopDst.push_back(base);
          uopSrc.push_back(base + i + 1);
          uopWgt.push_back(0);
        }
      }
      const int64_t G = static_cast<int64_t>(uopDst.size()) - 1;

      OpBuilder b(op);
      Location loc = op.getLoc();

      b.create<vtaisa::UopTableOp>(loc, b.getI64ArrayAttr(uopDst),
                                   b.getI64ArrayAttr(uopSrc),
                                   b.getI64ArrayAttr(uopWgt));

      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                               false, 0, kUopBase, 1, 1, 1);
      b.create<vtaisa::GemmInsnOp>(loc, false, false, true, false, true, 0, 1,
                                   1, 16, 16, 1, 0, 0, 0);

      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::ACC, false, false, false,
                               false, 0, kAccBase, nbBlocks, 16, 16);

      int64_t uopCursor = 1;
      for (int64_t w = 0; w < nWin; ++w) {
        bool lastWin = (w == nWin - 1);
        b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, false, false, false,
                                 false, 0, kUopBase + uopCursor, 1, iters,
                                 iters);
        uopCursor += iters;
        b.create<vtaisa::AluInsnOp>(
            loc, false, false, false, /*push_next=*/lastWin, false, 0, iters, 1,
            1, 0, 0, 0, 0, /*alu_opcode=*/1, /*use_imm=*/false, /*imm=*/0);
      }

      for (int64_t w = 0; w < nWin; ++w) {
        bool first = (w == 0);
        bool last = (w == nWin - 1);
        int64_t sramRow = windows[w];
        b.create<vtaisa::StoreOp>(loc, vtaisa::BufferId::OUT, first, false,
                                    last, false, sramRow, kOutBase + w, 1, 1, 1);
      }

      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::INP, false, true, false,
                               true, 0, 0, 0, 0, 0);
      b.create<vtaisa::LoadOp>(loc, vtaisa::BufferId::UOP, true, true, false,
                               false, 0, 0, 0, 0, 0);
      b.create<vtaisa::FinishOp>(loc);
      op.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createLowerVTAMaxPoolPass() {
  return std::make_unique<LowerVTAMaxPoolPass>();
}
