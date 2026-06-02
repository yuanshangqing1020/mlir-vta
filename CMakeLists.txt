cmake_minimum_required(VERSION 3.20)
project(mlir-vta LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${CMAKE_CURRENT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(include/mlir-vta/Dialect/VTA)
add_subdirectory(include/mlir-vta/Dialect/VTAISA)
add_subdirectory(lib/Dialect/VTA)
add_subdirectory(lib/Dialect/VTAISA)
add_subdirectory(lib/Transforms)
add_subdirectory(lib/Target)
add_subdirectory(tools/vta-opt)
add_subdirectory(tools/vta-translate)
