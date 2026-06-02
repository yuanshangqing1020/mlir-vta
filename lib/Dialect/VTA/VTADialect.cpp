#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir-vta/Dialect/VTA/VTAOps.h"
using namespace mlir;
using namespace mlir::vta;
#include "mlir-vta/Dialect/VTA/VTADialect.cpp.inc"
void VTADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-vta/Dialect/VTA/VTAOps.cpp.inc"
  >();
}
