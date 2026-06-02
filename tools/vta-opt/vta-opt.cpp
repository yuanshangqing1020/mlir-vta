#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Support/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::vta::VTADialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "vta-opt\n", registry));
}
