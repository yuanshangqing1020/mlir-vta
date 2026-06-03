// Rectangular GEMM entry: C[32x16] = A[32x48] * B[48x16] (Mb=2, Kb=3, Nb=1).
// Exercises the generalized multi-block lowering on non-square dims and the
// page-aligned DRAM allocation (WGT/ACC/OUT/UOP bases shift vs the 16/32 case).
// End-to-end acceptance: scripts/run_fsim_gemm.sh 32 48 16
func @main(%A: tensor<32x48xi32>, %B: tensor<48x16xi32>) -> tensor<32x16xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [32, 16] : tensor<32x16xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<32x16xi32> -> tensor<32x16xi32>
  %out = linalg.matmul ins(%A, %B : tensor<32x48xi32>, tensor<48x16xi32>)
                       outs(%acc : tensor<32x16xi32>) -> tensor<32x16xi32>
  return %out : tensor<32x16xi32>
}
