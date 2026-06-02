add_mlir_library(MLIRVTATarget
  VTABinaryEmitter.cpp
  DEPENDS VTAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRIR)
