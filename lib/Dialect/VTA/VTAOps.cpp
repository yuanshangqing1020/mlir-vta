#include "mlir-vta/Dialect/VTA/VTAOps.h"
#include "mlir-vta/Dialect/VTA/VTADialect.h"
#include "mlir/IR/OpImplementation.h"
using namespace mlir;
using namespace mlir::vta;
#define GET_OP_CLASSES
#include "mlir-vta/Dialect/VTA/VTAOps.cpp.inc"
