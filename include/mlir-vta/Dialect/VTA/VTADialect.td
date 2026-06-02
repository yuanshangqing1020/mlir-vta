#ifndef VTA_DIALECT_TD
#define VTA_DIALECT_TD
include "mlir/IR/OpBase.td"
def VTA_Dialect : Dialect {
  let name = "vta";
  let cppNamespace = "::mlir::vta";
  let summary = "VTA accelerator ISA dialect";
}
class VTA_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<VTA_Dialect, mnemonic, traits>;
#endif
