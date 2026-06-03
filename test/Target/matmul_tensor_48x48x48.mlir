// 3x3-block square GEMM entry: C[48x48] = A[48x48] * B[48x48].
// Exercises the merged LOAD y_size scaling (y=9) and per-block STORE count (9)
// plus the page-aligned DRAM allocation for a multi-page layer.
// End-to-end acceptance: scripts/run_fsim_gemm.sh 48 48 48
func @main(%A: tensor<48x48xi32>, %B: tensor<48x48xi32>) -> tensor<48x48xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [48, 48] : tensor<48x48xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<48x48xi32> -> tensor<48x48xi32>
  %out = linalg.matmul ins(%A, %B : tensor<48x48xi32>, tensor<48x48xi32>)
                       outs(%acc : tensor<48x48xi32>) -> tensor<48x48xi32>
  return %out : tensor<48x48xi32>
}
