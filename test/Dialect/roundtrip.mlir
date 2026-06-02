// RUN: vta-opt %s | vta-opt
module {
  "vta.uop_table"() {dst = [0, 0], src = [0, 0], wgt = [0, 0]} : () -> ()
  "vta.load"() {buffer_id = 0 : i64, dram_base = 5120 : i64, y_size = 1 : i64,
                x_size = 1 : i64, x_stride = 1 : i64, sram_base = 0 : i64} : () -> ()
  "vta.gemm_insn"() {reset = true, push_prev = true, uop_bgn = 0 : i64,
                     uop_end = 1 : i64, loop_out = 1 : i64, loop_in = 16 : i64,
                     dst_factor_out = 16 : i64, dst_factor_in = 1 : i64} : () -> ()
  "vta.finish"() : () -> ()
}
