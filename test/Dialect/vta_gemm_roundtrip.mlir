// Round-trips the custom ins(...)/outs(...) assembly format for vta.gemm.
module {
  %a = "test.src"() : () -> memref<16x16xi32>
  %b = "test.src"() : () -> memref<16x16xi32>
  %c = "test.src"() : () -> memref<16x16xi32>
  vta.gemm ins(%a, %b : memref<16x16xi32>, memref<16x16xi32>) outs(%c : memref<16x16xi32>)
}
