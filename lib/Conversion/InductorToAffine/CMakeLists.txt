add_mlir_library(InductorToAffine
	InductorToAffine.cpp
	#        LINK_LIBS PUBLIC
	#        InductorIR
	#        MLIRIR
	#        MLIRPass
	#        MLIRTransforms
        DEPENDS
	InductorAffineConversionPassIncGen
        )
