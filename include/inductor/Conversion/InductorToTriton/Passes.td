#ifndef INDUCTOR_CONVERSION_INDUCTOR_TO_TRITON_PASSES
#define INDUCTOR_CONVERSION_INDUCTOR_TO_TRITON_PASSES

include "mlir/Pass/PassBase.td"

def ConvertInductorToTriton: Pass<"convert-inductor-to-triton", "mlir::ModuleOp"> {
    let summary = "Convert Inductor to Triton";
    let description = [{

    }];
    let constructor = "::mlir::inductor::createConvertInductorToTritonPass()";

    let dependentDialects = ["mlir::affine::AffineDialect",
			    "mlir::arith::ArithDialect",
			    "mlir::math::MathDialect",
			    "mlir::scf::SCFDialect",
			    "mlir::triton::TritonDialect"];

   let options = [
       Option<"numWarps", "num-warps",
              "int32_t", /*default*/"4",
              "number of warps">,

       Option<"threadsPerWarp", "threads-per-warp",
              "int32_t", /*default*/"32",
              "number of threads per warp">,
        Option<"numCTAs", "num-ctas",
              "int32_t", /*default*/"1",
              "number of ctas in a cga">,
        Option<"computeCapability", "compute-capability",
              "int32_t", /*default*/"80",
              "compute capability">
   ];
}

#endif
