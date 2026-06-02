func @main(%A: memref<16x16xi32>, %B: memref<16x16xi32>, %C: memref<16x16xi32>) {
  "vta.gemm"(%A, %B, %C) : (memref<16x16xi32>, memref<16x16xi32>, memref<16x16xi32>) -> ()
  return
}
