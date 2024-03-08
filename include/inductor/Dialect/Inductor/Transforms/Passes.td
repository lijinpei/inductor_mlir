#ifndef INDUCTOR_PASSES
#define INDUCTOR_PASSES

include "mlir/Pass/PassBase.td"

def InductorGPUWrapperExtraction: Pass<"inductor-gpu-wrapper-extaction", "ModuleOp"> {
	let summary = "Extract a wrapper code for GPU device";
	let description = [{
		As GPU devices can not do tensor allocation. This pass will extract a wrapper that does scratch tensor allocation on the host-side, and launch device kernel with output/input/scatch tensors.
	}];
}

#endif
