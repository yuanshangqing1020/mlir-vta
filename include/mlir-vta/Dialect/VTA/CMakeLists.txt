set(LLVM_TARGET_DEFINITIONS VTADialect.td)
mlir_tablegen(VTADialect.h.inc -gen-dialect-decls -dialect=vta)
mlir_tablegen(VTADialect.cpp.inc -gen-dialect-defs -dialect=vta)
add_public_tablegen_target(VTADialectIncGen)
