#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/VTAGemmLayout.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Identifier.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

// Assigns page-aligned DRAM addresses to every vta.gemm "layer" in module
// order, threading a single cursor across layers so physical/logical bases
// climb monotonically (faithful to the upstream multi-layer dram_allocation).
// The result is recorded as a module attribute `vta.layers`: an array of
// dictionaries, one per layer, that lower-vta-gemm and the binary/data emitters
// consume. Logic is identical for a single layer (cursor starts at 0), so the
// existing single-layer goldens are unaffected whether or not this pass runs.
struct VTADramAllocationPass
    : public PassWrapper<VTADramAllocationPass, OperationPass<ModuleOp>> {
  StringRef getArgument() const final { return "vta-dram-allocation"; }
  StringRef getDescription() const final {
    return "Assign page-aligned DRAM addresses to vta.gemm layers (multi-layer "
           "base increment)";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<vta::GemmOp> layers;
    module.walk([&](vta::GemmOp g) { layers.push_back(g); });
    if (layers.empty())
      return;

    MLIRContext *ctx = module.getContext();
    Builder b(ctx);
    auto i64 = [&](int64_t v) { return b.getI64IntegerAttr(v); };
    auto id = [&](StringRef s) { return Identifier::get(s, ctx); };

    SmallVector<Attribute> layerAttrs;
    int64_t cursor = 0;
    for (vta::GemmOp g : layers) {
      auto lhsT = g.lhs().getType().dyn_cast<MemRefType>();
      auto rhsT = g.rhs().getType().dyn_cast<MemRefType>();
      if (!lhsT || !rhsT) {
        g.emitError("vta-dram-allocation: expected memref operands");
        signalPassFailure();
        return;
      }
      const int64_t Mb = lhsT.getShape()[0] / 16;
      const int64_t Kb = lhsT.getShape()[1] / 16;
      const int64_t Nb = rhsT.getShape()[1] / 16;

      vta::GemmLayout L = vta::computeGemmLayoutAt(Mb, Kb, Nb, cursor);
      cursor = L.lastPhys; // next layer continues from here

      StringRef name = g.name().getValueOr("");
      SmallVector<NamedAttribute> d;
      d.emplace_back(id("name"), b.getStringAttr(name));
      d.emplace_back(id("m"), i64(Mb * 16));
      d.emplace_back(id("k"), i64(Kb * 16));
      d.emplace_back(id("n"), i64(Nb * 16));
      d.emplace_back(id("inp_phys"), i64(L.inpPhys));
      d.emplace_back(id("wgt_phys"), i64(L.wgtPhys));
      d.emplace_back(id("acc_phys"), i64(L.accPhys));
      d.emplace_back(id("out_phys"), i64(L.outPhys));
      d.emplace_back(id("uop_phys"), i64(L.uopPhys));
      d.emplace_back(id("insn_phys"), i64(L.insnPhys));
      d.emplace_back(id("inp_log"), i64(L.inpLogical));
      d.emplace_back(id("wgt_log"), i64(L.wgtLogical));
      d.emplace_back(id("acc_log"), i64(L.accLogical));
      d.emplace_back(id("out_log"), i64(L.outLogical));
      d.emplace_back(id("uop_log"), i64(L.uopLogical));
      d.emplace_back(id("insn_log"), i64(L.insnLogical));
      d.emplace_back(id("last_phys"), i64(L.lastPhys));
      layerAttrs.push_back(b.getDictionaryAttr(d));
    }

    module->setAttr("vta.layers", b.getArrayAttr(layerAttrs));
  }
};

} // namespace

std::unique_ptr<Pass> mlir::vta::createVTADramAllocationPass() {
  return std::make_unique<VTADramAllocationPass>();
}
