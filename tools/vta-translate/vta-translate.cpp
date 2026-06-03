#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Target/VTABinaryEmitter.h"
#include "mlir-vta/Target/VTADataEmitter.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
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
  unsigned dataRows = 16, dataCols = 16, dataK = 16;
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
    } else if (arg == "--rows" && i + 1 < argc) {
      dataRows = std::stoul(argv[++i]);
    } else if (arg == "--cols" && i + 1 < argc) {
      dataCols = std::stoul(argv[++i]);
    } else if (arg == "--k" && i + 1 < argc) {
      dataK = std::stoul(argv[++i]);
    }
  }

  // Register all upstream dialects so the parser can handle whatever the
  // lowering pipeline emits (e.g. residual memref.alloc / linalg.fill from
  // bufferization), in addition to the VTA dialects whose ops we emit. The
  // emitter ignores non-vta/vtaisa ops, but they must still parse.
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::vta::VTADialect, mlir::vtaisa::VTAISADialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

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
    if (mlir::failed(mlir::vta::emitData(dataInput, dataWeight, outDir, dataRows,
                                         dataK, dataCols))) {
      llvm::errs() << "vta-translate: data emission failed\n";
      return 1;
    }
  }
  return 0;
}
