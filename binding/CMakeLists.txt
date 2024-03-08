declare_mlir_python_sources(InductorMLIRPythonSource)
declare_mlir_dialect_python_bindings(
	ADD_TO_PARENT InductorMLIRPythonSource
	ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/mlir"
	TD_FILE dialects/InductorOps.td
	SOURCES
	dialects/inductor.py
	DIALECT_NAME inductor
	GEN_ENUM_BINDINGS)

find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
declare_mlir_python_extension(InductorMLIRPythonExtension
	MODULE_NAME _mlirDialectsInductor
	ADD_TO_PARENT InductorMLIRPythonSource
	ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib"
  SOURCES
  DialectInductor.cpp
  PRIVATE_LINK_LIBS
  InductorIR
  MLIRInductorTransforms
  InductorToAffine
  PRIVATE_LINK_LIBS
  torch
  ${TORCH_PYTHON_LIBRARY}
)
add_mlir_python_modules(MLIRPythonModules
	ROOT_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/python_packages/inductor_mlir/mlir"
	INSTALL_PREFIX "python_packages/inductor_mlir/mlir"
  	DECLARED_SOURCES
  	InductorMLIRPythonSource
)
