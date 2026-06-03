add_mlir_library(MLIRVTATransforms
  LowerVTAGemm.cpp
  LowerVTAAlu.cpp
  ConvertLinalgToVTA.cpp
  VTADramAllocation.cpp
  VTASemaphoreDerive.cpp
  DEPENDS VTAOpsIncGen VTAISAOpsIncGen
  LINK_LIBS PUBLIC MLIRVTA MLIRVTAISA MLIRIR MLIRPass MLIRLinalg)
