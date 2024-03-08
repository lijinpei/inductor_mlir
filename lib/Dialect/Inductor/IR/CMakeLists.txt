add_mlir_dialect_library(InductorIR
	InductorDialect.cpp
	InductorOps.cpp

	DEPENDS
	MLIRInductorIncGen
	MLIRInductorAttrsIncGen

	ENABLE_AGGREGATION

	#LINK_LIBS PRIVATE
	#MLIRDialectUtils
	#MLIRIR
	#MLIRMathDialect
	#MLIRMemRefDialect
	#MLIRArithDialect
	#MLIRAffineDialect
)
