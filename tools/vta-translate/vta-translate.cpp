#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Target/VTABinaryEmitter.h"
#include "mlir-vta/Target/VTADataEmitter.h"

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: vta-translate <input.mlir> -o <outDir> "
                    "[--emit-data --input <inp.bin> --weight <wgt.bin>]\n";
    return 1;
  }

  std::string input = argv[1];
  std::string outDir = ".";
  bool emitData = false;
  std::string dataInput, dataWeight;
  for (int i = 2; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-o" && i + 1 < argc) {
      outDir = argv[++i];
    } else if (arg == "--emit-data") {
      emitData = true;
    } else if (arg == "--input" && i + 1 < argc) {
      dataInput = argv[++i];
    } else if (arg == "--weight" && i + 1 < argc) {
      dataWeight = argv[++i];
    }
  }

  mlir::MLIRContext context;
  // Standard dialect provides `std.return`, needed to parse func-wrapped IR
  // (FuncOp itself is builtin). The VTA dialects supply the ops we emit.
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::vta::VTADialect>();
  context.getOrLoadDialect<mlir::vtaisa::VTAISADialect>();

  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceFile<mlir::ModuleOp>(input, &context);
  if (!module) {
    llvm::errs() << "vta-translate: failed to parse " << input << "\n";
    return 1;
  }

  if (mlir::failed(mlir::vta::emitBinary(*module, outDir))) {
    llvm::errs() << "vta-translate: emission failed\n";
    return 1;
  }

  if (emitData) {
    if (dataInput.empty() || dataWeight.empty()) {
      llvm::errs() << "vta-translate: --emit-data requires --input and "
                      "--weight\n";
      return 1;
    }
    if (mlir::failed(mlir::vta::emitData(dataInput, dataWeight, outDir))) {
      llvm::errs() << "vta-translate: data emission failed\n";
      return 1;
    }
  }
  return 0;
}
