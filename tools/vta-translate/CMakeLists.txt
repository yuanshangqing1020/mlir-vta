add_llvm_executable(vta-translate vta-translate.cpp)
llvm_update_compile_flags(vta-translate)
target_link_libraries(vta-translate PRIVATE
  MLIRVTA MLIRVTATarget MLIRIR MLIRParser MLIRSupport)
mlir_check_all_link_libraries(vta-translate)
