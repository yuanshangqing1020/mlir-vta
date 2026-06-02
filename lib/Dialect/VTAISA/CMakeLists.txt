add_mlir_dialect_library(MLIRVTAISA
  VTAISADialect.cpp
  VTAISAOps.cpp
  DEPENDS VTAISADialectIncGen VTAISAOpsIncGen
  LINK_LIBS PUBLIC MLIRIR)
