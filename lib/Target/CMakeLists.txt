add_mlir_library(MLIRVTATarget
  VTABinaryEmitter.cpp
  VTADataEmitter.cpp
  DEPENDS VTAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRIR)
