#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "block GEMM: acc += lhs * rhs (in-place on acc memref), dims multiple of 16";
  // Optional `name` is the layer NAME (file suffix) for multi-layer compilation;
  // absent/"" means a single unnamed layer (instructions.bin without suffix).
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$acc,
                       OptionalAttr<StrAttr>:$name,
                       DefaultValuedAttr<I64Attr, "1">:$strategy);
  let results = (outs);
  let assemblyFormat =
    "`ins` `(` $lhs `,` $rhs `:` type($lhs) `,` type($rhs) `)` "
    "`outs` `(` $acc `:` type($acc) `)` attr-dict";
  let verifier = [{
    auto mt = [](::mlir::Value v) {
      return v.getType().dyn_cast<::mlir::MemRefType>();
    };
    auto lhsT = mt(lhs()), rhsT = mt(rhs()), accT = mt(acc());
    if (!lhsT || !rhsT || !accT || lhsT.getRank() != 2 ||
        rhsT.getRank() != 2 || accT.getRank() != 2)
      return emitOpError("vta.gemm requires three 2-D memref operands");
    int64_t m = lhsT.getShape()[0], k = lhsT.getShape()[1];
    int64_t k2 = rhsT.getShape()[0], n = rhsT.getShape()[1];
    int64_t am = accT.getShape()[0], an = accT.getShape()[1];
    // Phase-3: each dimension must be a positive multiple of 16.
    for (int64_t d : {m, k, n})
      if (d <= 0 || d % 16 != 0)
        return emitOpError("vta.gemm requires dimensions that are positive "
                           "multiples of 16");
    if (k2 != k || am != m || an != n)
      return emitOpError("vta.gemm shape mismatch: expected lhs MxK, rhs KxN, "
                         "acc MxN");
    int64_t s = strategy();
    if (s < 1 || s > 4)
      return emitOpError("vta.gemm strategy must be in [1,4], got ") << s;
    return ::mlir::success();
  }];
}

#endif
