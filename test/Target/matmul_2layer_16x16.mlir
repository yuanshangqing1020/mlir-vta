// Faithful two-layer compile: two independent 16x16 GEMM layers (L0, L1) in one
// func. -vta-dram-allocation threads a page-aligned cursor across them, so L1's
// logical bases climb (INP=448/WGT=32/ACC=576/OUT=640/UOP=11264) vs L0's
// (64/8/192/256/5120). Each layer lowers to its own finish-delimited stream;
// the binary emitter splits them into instructions<NAME>.bin / uop<NAME>.bin.
// Acceptance: scripts/run_fsim_2layer.sh ; golden test/golden/matmul_2layer_16x16/
func @main(%a0: memref<16x16xi32>, %b0: memref<16x16xi32>, %c0: memref<16x16xi32>,
           %a1: memref<16x16xi32>, %b1: memref<16x16xi32>, %c1: memref<16x16xi32>) {
  vta.gemm ins(%a0, %b0 : memref<16x16xi32>, memref<16x16xi32>) outs(%c0 : memref<16x16xi32>) {name = "L0"}
  vta.gemm ins(%a1, %b1 : memref<16x16xi32>, memref<16x16xi32>) outs(%c1 : memref<16x16xi32>) {name = "L1"}
  return
}
