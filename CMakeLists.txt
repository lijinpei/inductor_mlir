cmake_minimum_required(VERSION 3.28)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-fPIC ${CMAKE_CXX_FLAGS}")
# Note: upstream llvm/mlir header has problems with unused-parameter
# mlir tablegen generated file has problems with unused-local-typedefs and unused-function, at least temporarily.
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-local-typedefs -Wno-unused-function ${CMAKE_CXX_FLAGS}")

project(inductor_mlir)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(GNUInstallDirs)

find_package(LLVM REQUIRED CONFIG)
if(NOT MLIR_DIR)
  set(MLIR_DIR ${LLVM_LIBRARY_DIR}/cmake/mlir)
endif()
find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(AddMLIRPython)

include_directories(include)
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${PROJECT_BINARY_DIR}/include) # Tablegen'd files

include(handle_python)
include(handle_pytorch)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
include(handle_fmt)


add_subdirectory(include/inductor)
add_subdirectory(lib)
add_subdirectory(binding)
