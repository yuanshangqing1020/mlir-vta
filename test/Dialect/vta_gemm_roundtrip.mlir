// RUN: vta-opt %s | vta-opt | FileCheck %s --check-prefix=CHECK-CUSTOM
// CHECK-CUSTOM: vta.gemm ins({{.*}} : memref<16x16xi32>, memref<16x16xi32>) outs({{.*}} : memref<16x16xi32>)
// Round-trips the custom ins(...)/outs(...) assembly format for vta.gemm.
func @test(%a: memref<16x16xi32>, %b: memref<16x16xi32>, %c: memref<16x16xi32>) {
  vta.gemm ins(%a, %b : memref<16x16xi32>, memref<16x16xi32>) outs(%c : memref<16x16xi32>)
  return
}

// CHECK-CUSTOM: vta.gemm ins({{.*}} : memref<32x32xi32>, memref<32x32xi32>) outs({{.*}} : memref<32x32xi32>)
// Phase-3: multi-block dims (multiple of 16) also round-trip.
func @test_32x32(%a: memref<32x32xi32>, %b: memref<32x32xi32>, %c: memref<32x32xi32>) {
  vta.gemm ins(%a, %b : memref<32x32xi32>, memref<32x32xi32>) outs(%c : memref<32x32xi32>)
  return
}
