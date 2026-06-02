#pragma once
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace vta {

/// Walk the module body and emit byte-identical VTA binaries:
///   <outDir>/instructions.bin  (128-bit LE macro-instructions)
///   <outDir>/uop.bin           (32-bit LE micro-ops)
LogicalResult emitBinary(ModuleOp module, llvm::StringRef outDir);

} // namespace vta
} // namespace mlir
