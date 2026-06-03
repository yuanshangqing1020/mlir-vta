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
#include <vector>

int main(int argc, char **argv) {
  if (argc < 2) {
    llvm::errs() << "usage: vta-translate <input.mlir> -o <outDir> "
                    "[--emit-data --input <inp.bin> --weight <wgt.bin>]\n";
    return 1;
  }

  // One layer's data spec for multi-layer compilation.
  struct LayerSpec {
    std::string name, input, weight;
    unsigned rows = 16, k = 16, cols = 16;
  };

  std::string input = argv[1];
  std::string outDir = ".";
  bool emitData = false;
  std::string dataInput, dataWeight;
  unsigned dataRows = 16, dataCols = 16, dataK = 16;
  std::vector<LayerSpec> layers;
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
    } else if (arg == "--layer" && i + 1 < argc) {
      // --layer NAME=M,K,N,inputpath,weightpath
      std::string spec = argv[++i];
      auto eq = spec.find('=');
      if (eq == std::string::npos) {
        llvm::errs() << "vta-translate: --layer expects NAME=M,K,N,inp,wgt\n";
        return 1;
      }
      LayerSpec ls;
      ls.name = spec.substr(0, eq);
      std::string rest = spec.substr(eq + 1);
      std::vector<std::string> parts;
      size_t pos = 0;
      while (true) {
        auto comma = rest.find(',', pos);
        parts.push_back(rest.substr(pos, comma - pos));
        if (comma == std::string::npos)
          break;
        pos = comma + 1;
      }
      if (parts.size() != 5) {
        llvm::errs() << "vta-translate: --layer expects NAME=M,K,N,inp,wgt\n";
        return 1;
      }
      ls.rows = std::stoul(parts[0]);
      ls.k = std::stoul(parts[1]);
      ls.cols = std::stoul(parts[2]);
      ls.input = parts[3];
      ls.weight = parts[4];
      layers.push_back(ls);
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

  if (!layers.empty()) {
    // Multi-layer: emit per-layer data/metadata (suffixed). The binary emitter
    // already wrote the per-layer instructions/uop + layers_name/memory CSVs.
    for (const LayerSpec &ls : layers) {
      if (mlir::failed(mlir::vta::emitData(ls.input, ls.weight, outDir, ls.rows,
                                           ls.k, ls.cols, ls.name))) {
        llvm::errs() << "vta-translate: data emission failed for layer "
                     << ls.name << "\n";
        return 1;
      }
    }
  } else if (emitData) {
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
