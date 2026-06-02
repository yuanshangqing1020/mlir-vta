module  {
  func @main(%arg0: memref<16x16xi32>, %arg1: memref<16x16xi32>) -> memref<16x16xi32> {
    %c0_i32 = constant 0 : i32
    %0 = memref.alloc() : memref<16x16xi32>
    linalg.fill(%c0_i32, %0) : i32, memref<16x16xi32> 
    %1 = memref.alloc() : memref<16x16xi32>
    linalg.copy(%0, %1) : memref<16x16xi32>, memref<16x16xi32> 
    linalg.matmul ins(%arg0, %arg1 : memref<16x16xi32>, memref<16x16xi32>) outs(%1 : memref<16x16xi32>)
    return %1 : memref<16x16xi32>
  }
}

