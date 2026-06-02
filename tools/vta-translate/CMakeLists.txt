get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(vta-translate vta-translate.cpp)
llvm_update_compile_flags(vta-translate)
target_link_libraries(vta-translate PRIVATE
  ${dialect_libs} ${conversion_libs}
  MLIRVTA MLIRVTAISA MLIRVTATarget MLIRIR MLIRParser MLIRSupport MLIRStandard)
mlir_check_all_link_libraries(vta-translate)
