#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "High-level VTA GEMM op (single tile)";
  let arguments = (ins
    I64Attr:$m,
    I64Attr:$n,
    I64Attr:$k
  );
  let assemblyFormat = "attr-dict";
}

#endif
