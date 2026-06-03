#pragma once
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace vta {

/// Emit the block-layout data files and metadata CSVs for a (multiple-of-16)
/// GEMM `C[MxN] = A[MxK] * B[KxN]`, byte-identical to the Python VTA compiler's
/// compiler_output/:
///   <outDir>/input.bin          (M*K*4 B, block-major, no transpose)
///   <outDir>/weight.bin         (K*N*4 B, block-major, each 16x16 block transposed)
///   <outDir>/accumulator.bin    ((M/16)*(N/16)*1024 B, zeros)
///   <outDir>/out_init.bin       (same size, zeros)
///   <outDir>/metadata.csv
///   <outDir>/memory_addresses.csv
///   <outDir>/layers_name.csv
///
/// `inputPath` holds the raw MxK row-major int32 matrix; `weightPath` the raw
/// KxN row-major int32 matrix. Dimensions default to a single 16x16 block.
LogicalResult emitData(llvm::StringRef inputPath, llvm::StringRef weightPath,
                       llvm::StringRef outDir, unsigned mRows = 16,
                       unsigned kDim = 16, unsigned nCols = 16);

} // namespace vta
} // namespace mlir
