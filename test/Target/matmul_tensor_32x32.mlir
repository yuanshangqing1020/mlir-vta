// Phase-3 entry: tensor-level 32x32 linalg.matmul. No linalg-tile; the block
// schedule lives inside lower-vta-gemm. Uses std-style `constant` because this
// build's vta-opt does not register the arith dialect.
func @main(%A: tensor<32x32xi32>, %B: tensor<32x32xi32>) -> tensor<32x32xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [32, 32] : tensor<32x32xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<32x32xi32> -> tensor<32x32xi32>
  %out = linalg.matmul ins(%A, %B : tensor<32x32xi32>, tensor<32x32xi32>)
                       outs(%acc : tensor<32x32xi32>) -> tensor<32x32xi32>
  return %out : tensor<32x32xi32>
}
