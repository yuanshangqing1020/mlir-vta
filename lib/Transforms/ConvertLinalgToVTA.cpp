#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

struct ConvertLinalgToVTAPass
    : public PassWrapper<ConvertLinalgToVTAPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "convert-linalg-to-vta"; }
  StringRef getDescription() const final {
    return "Convert bufferized 16x16 linalg.matmul to vta.gemm";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vta::VTADialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<linalg::MatmulOp> targets;
    module.walk([&](linalg::MatmulOp m) { targets.push_back(m); });

    for (linalg::MatmulOp m : targets) {
      if (!m.hasBufferSemantics()) {
        m.emitError("convert-linalg-to-vta: expected bufferized (memref) "
                    "linalg.matmul; run bufferization first");
        signalPassFailure();
        return;
      }
      Value lhs = m.inputs()[0];
      Value rhs = m.inputs()[1];
      Value acc = m.outputs()[0];

      OpBuilder b(m);
      // Single unnamed layer (no `name` attribute); multi-layer naming is set
      // explicitly on hand-written vta.gemm ops.
      b.create<vta::GemmOp>(m.getLoc(), lhs, rhs, acc, /*name=*/StringAttr());
      m.erase();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createConvertLinalgToVTAPass() {
  return std::make_unique<ConvertLinalgToVTAPass>();
}
