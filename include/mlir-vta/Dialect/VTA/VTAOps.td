#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "single 16x16 block GEMM: acc += lhs * rhs (in-place on acc memref)";
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$acc);
  let results = (outs);
  let assemblyFormat =
    "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
    "`outs` `(` $acc `:` type($acc) `)` attr-dict";
  let verifier = [{
    auto isBlk = [](::mlir::Value v) {
      auto mt = v.getType().dyn_cast<::mlir::MemRefType>();
      return mt && mt.getRank() == 2 && mt.getShape()[0] == 16 &&
             mt.getShape()[1] == 16;
    };
    if (!isBlk(lhs()) || !isBlk(rhs()) || !isBlk(acc()))
      return emitOpError("phase-2 vta.gemm requires 16x16 memref operands");
    return ::mlir::success();
  }];
}

#endif
