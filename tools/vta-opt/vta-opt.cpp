#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAPasses.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::vta::VTADialect, mlir::vtaisa::VTAISADialect>();
  mlir::vta::registerVTAPasses();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "vta-opt\n", registry));
}
