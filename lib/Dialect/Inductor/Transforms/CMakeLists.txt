add_mlir_dialect_library(MLIRInductorTransforms
	GPUWrapperExtraction.cpp

	DEPENDS
	MLIRInductorTransformsIncGen
	LINK_LIBS
	PRIVATE fmt::fmt-header-only
  #  LINK_LIBS PUBLIC
  #  MLIRAnalysis
  #  MLIRArithDialect
  #  MLIRBufferizationDialect
  #  MLIRBufferizationTransforms
  #  MLIRFuncDialect
  #  MLIRFuncTransforms
  #  MLIRInferIntRangeInterface
  #  MLIRIR
  #  MLIRMemRefDialect
  #  MLIRPass
  #  MLIRTensorDialect
  #  MLIRTransforms
  #  MLIRTransformUtils
  #  MLIRValueBoundsOpInterface
  #  MLIRVectorDialect
  )
