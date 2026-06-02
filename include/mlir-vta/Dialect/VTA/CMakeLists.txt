set(LLVM_TARGET_DEFINITIONS VTADialect.td)
mlir_tablegen(VTADialect.h.inc -gen-dialect-decls -dialect=vta)
mlir_tablegen(VTADialect.cpp.inc -gen-dialect-defs -dialect=vta)
add_public_tablegen_target(VTADialectIncGen)

set(LLVM_TARGET_DEFINITIONS VTAOps.td)
mlir_tablegen(VTAOps.h.inc -gen-op-decls)
mlir_tablegen(VTAOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(VTAOpsIncGen)
