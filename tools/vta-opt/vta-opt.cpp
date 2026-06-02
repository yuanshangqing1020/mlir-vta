#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  registry.insert<mlir::vta::VTADialect, mlir::vtaisa::VTAISADialect>();
  mlir::vta::registerVTAPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "vta-opt\n", registry));
}
