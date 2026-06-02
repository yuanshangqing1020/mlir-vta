add_mlir_dialect_library(MLIRVTA
  VTADialect.cpp
  DEPENDS VTADialectIncGen
  LINK_LIBS PUBLIC MLIRIR)
