module {
  "vtaisa.uop_table"() {dst = [0, 0], src = [0, 0], wgt = [0, 0]} : () -> ()
  "vtaisa.load"() {buffer_id = 0 : i64, sram_base = 0 : i64, dram_base = 5120 : i64, y_size = 1 : i64, x_size = 1 : i64, x_stride = 1 : i64} : () -> ()
  "vtaisa.gemm_insn"() {push_prev = true, reset = true, uop_bgn = 0 : i64, uop_end = 1 : i64, loop_out = 1 : i64, loop_in = 16 : i64, dst_factor_out = 16 : i64, dst_factor_in = 1 : i64} : () -> ()
  "vtaisa.load"() {pop_next = true, buffer_id = 2 : i64, sram_base = 0 : i64, dram_base = 64 : i64, y_size = 1 : i64, x_size = 16 : i64, x_stride = 16 : i64} : () -> ()
  "vtaisa.load"() {push_next = true, buffer_id = 1 : i64, sram_base = 0 : i64, dram_base = 8 : i64, y_size = 1 : i64, x_size = 1 : i64, x_stride = 1 : i64} : () -> ()
  "vtaisa.load"() {pop_prev = true, buffer_id = 3 : i64, sram_base = 0 : i64, dram_base = 192 : i64, y_size = 1 : i64, x_size = 16 : i64, x_stride = 16 : i64} : () -> ()
  "vtaisa.load"() {buffer_id = 0 : i64, sram_base = 0 : i64, dram_base = 5121 : i64, y_size = 1 : i64, x_size = 1 : i64, x_stride = 1 : i64} : () -> ()
  "vtaisa.gemm_insn"() {push_prev = true, push_next = true, uop_bgn = 0 : i64, uop_end = 1 : i64, loop_out = 1 : i64, loop_in = 16 : i64, dst_factor_in = 1 : i64, src_factor_in = 1 : i64} : () -> ()
  "vtaisa.store"() {pop_prev = true, push_prev = true, buffer_id = 4 : i64, sram_base = 0 : i64, dram_base = 256 : i64, y_size = 1 : i64, x_size = 16 : i64, x_stride = 16 : i64} : () -> ()
  "vtaisa.load"() {pop_next = true, push_next = true, buffer_id = 2 : i64, sram_base = 0 : i64, dram_base = 0 : i64, y_size = 0 : i64, x_size = 0 : i64, x_stride = 0 : i64} : () -> ()
  "vtaisa.load"() {pop_prev = true, pop_next = true, buffer_id = 0 : i64, sram_base = 0 : i64, dram_base = 0 : i64, y_size = 0 : i64, x_size = 0 : i64, x_stride = 0 : i64} : () -> ()
  "vtaisa.finish"() : () -> ()
}
