module {
  %a = "test.src"() : () -> memref<8x8xi32>
  %b = "test.src"() : () -> memref<8x8xi32>
  %c = "test.src"() : () -> memref<8x8xi32>
  vta.gemm ins(%a, %b : memref<8x8xi32>, memref<8x8xi32>) outs(%c : memref<8x8xi32>)
}
