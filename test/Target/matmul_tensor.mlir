func @main(%A: tensor<16x16xi32>, %B: tensor<16x16xi32>) -> tensor<16x16xi32> {
  %c0 = constant 0 : i32
  %init = linalg.init_tensor [16, 16] : tensor<16x16xi32>
  %acc = linalg.fill(%c0, %init) : i32, tensor<16x16xi32> -> tensor<16x16xi32>
  %out = linalg.matmul
           ins(%A, %B : tensor<16x16xi32>, tensor<16x16xi32>)
           outs(%acc : tensor<16x16xi32>) -> tensor<16x16xi32>
  return %out : tensor<16x16xi32>
}
