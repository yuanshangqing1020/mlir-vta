#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VTA_GemmOp : VTA_Op<"gemm"> {
  let summary = "block GEMM: acc += lhs * rhs (in-place on acc memref); K/N multiples of 16";
  // Optional `name` is the layer NAME (file suffix) for multi-layer compilation;
  // absent/"" means a single unnamed layer (instructions.bin without suffix).
  // `expand_bias`: when true, ACC is a 1xN row broadcast at VTA compile time
  // (QLinearConv bias), matching upstream doExpandBias.
  let arguments = (ins AnyMemRef:$lhs, AnyMemRef:$rhs, AnyMemRef:$acc,
                       OptionalAttr<StrAttr>:$name,
                       DefaultValuedAttr<I64Attr, "1">:$strategy,
                       DefaultValuedAttr<BoolAttr, "false">:$expand_bias);
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
    // M may be any positive row count (partial 16-row blocks); K/N must be
    // positive multiples of 16 (column blocking).
    if (m <= 0)
      return emitOpError("vta.gemm requires positive M rows");
    for (int64_t d : {k, n})
      if (d <= 0 || d % 16 != 0)
        return emitOpError("vta.gemm requires K/N that are positive multiples "
                           "of 16");
    if (k2 != k || am != m || an != n)
      return emitOpError("vta.gemm shape mismatch: expected lhs MxK, rhs KxN, "
                         "acc MxN");
    int64_t s = strategy();
    if (s < 1 || s > 4)
      return emitOpError("vta.gemm strategy must be in [1,4], got ") << s;
    return ::mlir::success();
  }];
}

// ALU opcode constants (must match vtaisa.alu_insn alu_opcode field and
// upstream VTA ISA: MIN=0 MAX=1 ADD=2 SHR=3 MUL=4).
def VTA_AluOp : VTA_Op<"alu"> {
  let summary = "element-wise ALU on acc memref: C = op(acc, imm) or op(acc, src)";
  // `acc` is the input/output memref (both ACC and OUT role):
  //   shape [M, 16] where M rows = number of 16-element vectors.
  //   M must be a positive multiple of 1 (≥1), 16 elements wide.
  // `alu_opcode`: 0=MIN,1=MAX,2=ADD,3=SHR,4=MUL
  // `use_imm`: true → operand is the signed immediate `imm`; false → inter-vector
  // `imm`: immediate value (for ADD_IMM: bias shift, e.g. 3)
  let arguments = (ins AnyMemRef:$acc,
                       I64Attr:$alu_opcode,
                       DefaultValuedAttr<BoolAttr, "false">:$use_imm,
                       DefaultValuedAttr<I64Attr, "0">:$imm,
                       OptionalAttr<StrAttr>:$name);
  let results = (outs);
  let assemblyFormat =
    "`ins` `(` $acc `:` type($acc) `)` attr-dict";
  let verifier = [{
    auto accT = acc().getType().dyn_cast<::mlir::MemRefType>();
    if (!accT || accT.getRank() != 2)
      return emitOpError("vta.alu requires a 2-D memref operand");
    int64_t rows = accT.getShape()[0];
    int64_t cols = accT.getShape()[1];
    if (rows <= 0 || cols != 16)
      return emitOpError("vta.alu acc memref must have shape [M, 16] (M rows, 16 cols)");
    int64_t op = alu_opcode();
    if (op < 0 || op > 4)
      return emitOpError("vta.alu alu_opcode must be in [0,4], got ") << op;
    return ::mlir::success();
  }];
}

#endif
