#ifndef INDUCTOR_CONVERSION_INDUCTOR_TO_AFFINE_PASSES
#define INDUCTOR_CONVERSION_INDUCTOR_TO_AFFINE_PASSES

include "mlir/Pass/PassBase.td"

def ConvertInductorToAffine : Pass<"convert-inductor-to-affine", "mlir::ModuleOp"> {
    let summary = "Convert Inductor to Affine";
    let description = [{

    }];
    let constructor = "::mlir::inductor::createConvertInductorToAffinePass()";

    let dependentDialects = ["mlir::affine::AffineDialect",
    	"mlir::index::IndexDialect",
    ];
}

#endif
