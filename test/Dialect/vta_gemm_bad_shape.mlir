// RUN: not vta-opt %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: phase-2 vta.gemm requires 16x16 memref operands
func @test(%a: memref<8x8xi32>, %b: memref<8x8xi32>, %c: memref<8x8xi32>) {
  vta.gemm ins(%a, %b : memref<8x8xi32>, memref<8x8xi32>) outs(%c : memref<8x8xi32>)
  return
}
