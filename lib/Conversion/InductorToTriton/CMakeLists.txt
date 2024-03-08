include_directories(${PROJECT_BINARY_DIR}/third_party/triton/include)
include_directories(${PROJECT_SOURCE_DIR}/third_party/triton/include)

add_mlir_library(InductorToTriton
	InductorToTriton.cpp
	LINK_LIBS PUBLIC
	InductorIR
	MLIRIR
	MLIRPass
	MLIRTransforms
	DEPENDS
	InductorConversionPassIncGen
	)
