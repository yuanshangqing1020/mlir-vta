#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Target/VTABinaryEmitter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: vta-translate <input.mlir> -o <outDir>\n";
    return 1;
  }

  std::string input = argv[1];
  std::string outDir = ".";
  for (int i = 2; i < argc; ++i) {
    if (std::string(argv[i]) == "-o" && i + 1 < argc) {
      outDir = argv[i + 1];
      ++i;
    }
  }

  mlir::MLIRContext context;
  context.getOrLoadDialect<mlir::vta::VTADialect>();

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
  return 0;
}
