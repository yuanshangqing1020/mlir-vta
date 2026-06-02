add_mlir_library(MLIRVTATarget
  VTABinaryEmitter.cpp
  VTADataEmitter.cpp
  DEPENDS VTAISAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTAISA MLIRIR)
