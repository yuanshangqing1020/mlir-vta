#include "mlir-vta/Dialect/VTAISA/VTAISAOps.h"
#include "mlir-vta/Dialect/VTAISA/VTAISADialect.h"
#include "mlir/IR/OpImplementation.h"
using namespace mlir;
using namespace mlir::vtaisa;
#include "mlir-vta/Dialect/VTAISA/VTAISAEnums.cpp.inc"
#define GET_OP_CLASSES
#include "mlir-vta/Dialect/VTAISA/VTAISAOps.cpp.inc"
