// Phase-3 overfit entry: M=16, K=2064, N=16 (strategy-1, 2 steps)
// Triggers overfit because nbA=129 >= 128 (INP buffer capacity).
// No linalg-tile; the multi-step schedule lives inside lower-vta-gemm.
func @main(%A: tensor<16x2064xi32>, %B: tensor<2064x16xi32>) -> tensor<16x16xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<16x16xi32> -> tensor<16x16xi32>
  %out = linalg.matmul ins(%A, %B : tensor<16x2064xi32>, tensor<2064x16xi32>)
                       outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %out : tensor<16x16xi32>
}
