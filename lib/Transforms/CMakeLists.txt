add_mlir_library(MLIRVTATransforms
  LowerVTAGemm.cpp
  DEPENDS VTAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRIR MLIRPass)
