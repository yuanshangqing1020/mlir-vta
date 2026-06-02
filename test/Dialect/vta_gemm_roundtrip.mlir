// RUN: vta-opt %s | vta-opt | FileCheck %s --check-prefix=CHECK-CUSTOM
// CHECK-CUSTOM: vta.gemm ins({{.*}} : memref<16x16xi32>, memref<16x16xi32>) outs({{.*}} : memref<16x16xi32>)
// Round-trips the custom ins(...)/outs(...) assembly format for vta.gemm.
func @test(%a: memref<16x16xi32>, %b: memref<16x16xi32>, %c: memref<16x16xi32>) {
  vta.gemm ins(%a, %b : memref<16x16xi32>, memref<16x16xi32>) outs(%c : memref<16x16xi32>)
  return
}
