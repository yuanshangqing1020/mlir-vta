#ifndef VTAISA_ENUMS_TD
#define VTAISA_ENUMS_TD
include "mlir/IR/OpBase.td"
def VTAISA_BufUOP : I64EnumAttrCase<"UOP", 0>;
def VTAISA_BufWGT : I64EnumAttrCase<"WGT", 1>;
def VTAISA_BufINP : I64EnumAttrCase<"INP", 2>;
def VTAISA_BufACC : I64EnumAttrCase<"ACC", 3>;
def VTAISA_BufOUT : I64EnumAttrCase<"OUT", 4>;
def VTAISA_BufferId : I64EnumAttr<"BufferId", "VTA on-chip buffer",
    [VTAISA_BufUOP, VTAISA_BufWGT, VTAISA_BufINP, VTAISA_BufACC, VTAISA_BufOUT]> {
  let cppNamespace = "::mlir::vtaisa";
}
#endif
