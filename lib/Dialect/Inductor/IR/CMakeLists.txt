add_mlir_dialect_library(InductorIR
	InductorDialect.cpp
	InductorOps.cpp
	InductorInterfaces.cpp

	DEPENDS
	MLIRInductorIncGen
	MLIRInductorAttrsIncGen

	LINK_LIBS PUBLIC
	MLIRDialectUtils
	MLIRIR
	MLIRMathDialect
	MLIRMemRefDialect
	MLIRArithDialect
	MLIRAffineDialect
)
