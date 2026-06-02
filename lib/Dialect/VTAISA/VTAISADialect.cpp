#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
using namespace mlir;
using namespace mlir::vtaisa;
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.cpp.inc"
void VTAISADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.cpp.inc"
  >();
}
