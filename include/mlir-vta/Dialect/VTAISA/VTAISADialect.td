#ifndef VTAISA_DIALECT_TD
#define VTAISA_DIALECT_TD
include "mlir/IR/OpBase.td"
def VTAISA_Dialect : Dialect {
  let name = "vtaisa";
  let cppNamespace = "::mlir::vtaisa";
  let summary = "VTA low-level ISA dialect (1:1 with 128-bit macro-instructions / 32-bit UOPs)";
}
class VTAISA_Op<string mnemonic, list<OpTrait> traits = []>
    : Op<VTAISA_Dialect, mnemonic, traits>;
#endif
