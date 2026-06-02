#pragma once
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace vta {

/// Emit the block-layout data files and metadata CSVs for the 16x16 GEMM case,
/// byte-identical to the Python VTA compiler's compiler_output/:
///   <outDir>/input.bin          (1024 B, raw copy of the 16x16 int32 input)
///   <outDir>/weight.bin         (1024 B, transpose of the 16x16 int32 weight)
///   <outDir>/accumulator.bin    (1024 B, zeros)
///   <outDir>/out_init.bin       (1024 B, zeros)
///   <outDir>/metadata.csv
///   <outDir>/memory_addresses.csv
///   <outDir>/layers_name.csv
LogicalResult emitData(llvm::StringRef inputPath, llvm::StringRef weightPath,
                       llvm::StringRef outDir);

} // namespace vta
} // namespace mlir
