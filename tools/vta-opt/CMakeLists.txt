get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(vta-opt vta-opt.cpp)
llvm_update_compile_flags(vta-opt)
target_link_libraries(vta-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  MLIRVTA MLIRVTAISA MLIRVTATransforms
  MLIROptLib MLIRIR MLIRParser MLIRSupport)
mlir_check_all_link_libraries(vta-opt)
