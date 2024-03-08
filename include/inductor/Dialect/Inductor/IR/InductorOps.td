#ifndef INDUCTOR_OPS
#define INDUCTOR_OPS

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/CommonTypeConstraints.td"

include "inductor/Dialect/Inductor/IR/InductorBase.td"
include "inductor/Dialect/Inductor/IR/InductorAttrs.td"

class Inductor_Op<string mnemonic, list<Trait> traits = []>
	: Op<Inductor_Dialect, mnemonic, traits> {
}

def Inductor_PointwiseLoopOp : Inductor_Op<"pointwise_loop", []> {
	let arguments = (ins AnyMemRef:$buffer, Variadic<Index>:$ranges);
	let regions = (region AnyRegion:$inner_fn);
}

def Inductor_ReductionLoopOp : Inductor_Op<"reduction_loop", [AttrSizedOperandSegments]> {
	let arguments = (ins AnyMemRef:$buffer, Variadic<Index>:$ranges, Variadic<Index>:$reduction_ranges, InductorReductionTypeAttr:$reduction_type, InductorReductionHintAttr:$reduction_hint);
	let regions = (region AnyRegion:$inner_fn);
}

def Inductor_LoopYieldOp: Inductor_Op<"loopYield", [Terminator]> {
	let arguments = (ins AnyType:$value);
}

#endif // INDUCTOR_OPS
