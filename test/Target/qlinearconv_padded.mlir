// Padded QLinearConv GEMM (increment A): 25x27 x 27x3 -> 25x3
// mlir dims 32x32x16 (rows padded to 16-multiple for vta.gemm verifier)
func @main(%lhs: memref<32x32xi32>, %rhs: memref<32x16xi32>, %acc: memref<32x16xi32>) {
  vta.gemm ins(%lhs, %rhs : memref<32x32xi32>, memref<32x16xi32>)
           outs(%acc : memref<32x16xi32>) {name = "QLinearConv1", strategy = 2 : i64}
  return
}
