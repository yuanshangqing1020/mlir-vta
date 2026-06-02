add_mlir_dialect_library(MLIRVTA
  VTADialect.cpp
  VTAOps.cpp
  DEPENDS VTADialectIncGen VTAOpsIncGen
  LINK_LIBS PUBLIC MLIRIR)
