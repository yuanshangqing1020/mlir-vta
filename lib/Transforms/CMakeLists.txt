add_mlir_library(MLIRVTATransforms
  LowerVTAGemm.cpp
  ConvertLinalgToVTA.cpp
  DEPENDS VTAOpsIncGen VTAISAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRVTAISA MLIRIR MLIRPass MLIRLinalg)
