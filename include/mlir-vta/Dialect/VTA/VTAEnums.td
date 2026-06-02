#ifndef VTA_ENUMS_TD
#define VTA_ENUMS_TD
include "mlir/IR/OpBase.td"
def VTA_BufUOP : I64EnumAttrCase<"UOP", 0>;
def VTA_BufWGT : I64EnumAttrCase<"WGT", 1>;
def VTA_BufINP : I64EnumAttrCase<"INP", 2>;
def VTA_BufACC : I64EnumAttrCase<"ACC", 3>;
def VTA_BufOUT : I64EnumAttrCase<"OUT", 4>;
def VTA_BufferId : I64EnumAttr<"BufferId", "VTA on-chip buffer",
    [VTA_BufUOP, VTA_BufWGT, VTA_BufINP, VTA_BufACC, VTA_BufOUT]> {
  let cppNamespace = "::mlir::vta";
}
#endif
