#ifndef VTA_OPS_TD
#define VTA_OPS_TD
include "mlir-vta/Dialect/VTA/VTADialect.td"
include "mlir-vta/Dialect/VTA/VTAEnums.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def VTA_LoadOp : VTA_Op<"load"> {
  let summary = "VTA LOAD instruction";
  let arguments = (ins
    VTA_BufferId:$buffer_id,
    DefaultValuedAttr<BoolAttr, "false">:$pop_prev,
    DefaultValuedAttr<BoolAttr, "false">:$pop_next,
    DefaultValuedAttr<BoolAttr, "false">:$push_prev,
    DefaultValuedAttr<BoolAttr, "false">:$push_next,
    I64Attr:$sram_base,
    I64Attr:$dram_base,
    I64Attr:$y_size,
    I64Attr:$x_size,
    I64Attr:$x_stride,
    DefaultValuedAttr<I64Attr, "0">:$y_pad_top,
    DefaultValuedAttr<I64Attr, "0">:$y_pad_bottom,
    DefaultValuedAttr<I64Attr, "0">:$x_pad_left,
    DefaultValuedAttr<I64Attr, "0">:$x_pad_right
  );
  let assemblyFormat = "attr-dict";
}

def VTA_StoreOp : VTA_Op<"store"> {
  let summary = "VTA STORE instruction";
  let arguments = (ins
    VTA_BufferId:$buffer_id,
    DefaultValuedAttr<BoolAttr, "false">:$pop_prev,
    DefaultValuedAttr<BoolAttr, "false">:$pop_next,
    DefaultValuedAttr<BoolAttr, "false">:$push_prev,
    DefaultValuedAttr<BoolAttr, "false">:$push_next,
    I64Attr:$sram_base,
    I64Attr:$dram_base,
    I64Attr:$y_size,
    I64Attr:$x_size,
    I64Attr:$x_stride
  );
  let assemblyFormat = "attr-dict";
}

def VTA_GemmInsnOp : VTA_Op<"gemm_insn"> {
  let summary = "VTA GEMM micro-op loop instruction";
  let arguments = (ins
    DefaultValuedAttr<BoolAttr, "false">:$pop_prev,
    DefaultValuedAttr<BoolAttr, "false">:$pop_next,
    DefaultValuedAttr<BoolAttr, "false">:$push_prev,
    DefaultValuedAttr<BoolAttr, "false">:$push_next,
    DefaultValuedAttr<BoolAttr, "false">:$reset,
    I64Attr:$uop_bgn,
    I64Attr:$uop_end,
    I64Attr:$loop_out,
    I64Attr:$loop_in,
    DefaultValuedAttr<I64Attr, "0">:$dst_factor_out,
    DefaultValuedAttr<I64Attr, "0">:$dst_factor_in,
    DefaultValuedAttr<I64Attr, "0">:$src_factor_out,
    DefaultValuedAttr<I64Attr, "0">:$src_factor_in,
    DefaultValuedAttr<I64Attr, "0">:$wgt_factor_out,
    DefaultValuedAttr<I64Attr, "0">:$wgt_factor_in
  );
  let assemblyFormat = "attr-dict";
}

def VTA_AluInsnOp : VTA_Op<"alu_insn"> {
  let summary = "VTA ALU micro-op loop instruction";
  let arguments = (ins
    DefaultValuedAttr<BoolAttr, "false">:$pop_prev,
    DefaultValuedAttr<BoolAttr, "false">:$pop_next,
    DefaultValuedAttr<BoolAttr, "false">:$push_prev,
    DefaultValuedAttr<BoolAttr, "false">:$push_next,
    DefaultValuedAttr<BoolAttr, "false">:$reset,
    I64Attr:$uop_bgn,
    I64Attr:$uop_end,
    I64Attr:$loop_out,
    I64Attr:$loop_in,
    DefaultValuedAttr<I64Attr, "0">:$dst_factor_out,
    DefaultValuedAttr<I64Attr, "0">:$dst_factor_in,
    DefaultValuedAttr<I64Attr, "0">:$src_factor_out,
    DefaultValuedAttr<I64Attr, "0">:$src_factor_in,
    I64Attr:$alu_opcode,
    DefaultValuedAttr<BoolAttr, "false">:$use_imm,
    DefaultValuedAttr<I64Attr, "0">:$imm
  );
  let assemblyFormat = "attr-dict";
}

def VTA_FinishOp : VTA_Op<"finish"> {
  let summary = "VTA FINISH instruction";
  let arguments = (ins);
  let assemblyFormat = "attr-dict";
}

def VTA_UopTableOp : VTA_Op<"uop_table"> {
  let summary = "VTA micro-op table";
  let arguments = (ins
    I64ArrayAttr:$dst,
    I64ArrayAttr:$src,
    I64ArrayAttr:$wgt
  );
  let assemblyFormat = "attr-dict";
}

#endif
