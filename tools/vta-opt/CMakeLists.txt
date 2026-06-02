add_llvm_executable(vta-opt vta-opt.cpp)
llvm_update_compile_flags(vta-opt)
target_link_libraries(vta-opt PRIVATE MLIRVTA MLIRVTAISA MLIRVTATransforms MLIROptLib MLIRIR MLIRParser MLIRSupport)
mlir_check_all_link_libraries(vta-opt)
