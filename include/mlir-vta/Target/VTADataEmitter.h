#pragma once
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace vta {

/// Emit block-layout data files for GEMM C[MxN] = A[MxK] * B[KxN] + bias.
/// M may be any positive row count; K/N must be multiples of 16.
/// When `accPath` is non-empty, read a row-major int32 bias/accumulator matrix
/// of shape accRows x nCols (typically 1 x N for QLinearConv expand_bias).
LogicalResult emitData(llvm::StringRef inputPath, llvm::StringRef weightPath,
                       llvm::StringRef outDir, unsigned mRows = 16,
                       unsigned kDim = 16, unsigned nCols = 16,
                       llvm::StringRef name = "",
                       llvm::StringRef accPath = "", unsigned accRows = 0);

} // namespace vta
} // namespace mlir
