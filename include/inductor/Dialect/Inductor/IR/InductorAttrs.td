#ifndef INDUCTOR_ATTRIBUTES
#define INDUCTOR_ATTRIBUTES

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"

include "inductor/Dialect/Inductor/IR/InductorBase.td"

class Inductor_Attr<string name, string mnem>
	: AttrDef<Inductor_Dialect, name> {
	let mnemonic = mnem;
}

def InductorDeviceAttr : Inductor_Attr<"Device", "device"> {
	let parameters = (ins Builtin_StringAttr:$device);
	let assemblyFormat = "$device";
}

def InductorReductionHintEnum : I32EnumAttr<"ReductionHint", "", [
I32EnumAttrCase<"INNER", 0>,
I32EnumAttrCase<"OUTER", 1>,
I32EnumAttrCase<"OUTER_TINY", 2>,
I32EnumAttrCase<"DEFAULT", 3>,
]> {
	let genSpecializedAttr = 0;
	let cppNamespace = "::mlir::inductor";
}
def InductorReductionHintAttr : EnumAttr<Inductor_Dialect, InductorReductionHintEnum, "reduction_hint">;

def InductorLayoutEnum : I32EnumAttr<"Layout", "", [
I32EnumAttrCase<"Fixed", 0>,
I32EnumAttrCase<"Flexible", 1>,
]> {
	let genSpecializedAttr = 0;
	let cppNamespace = "::mlir::inductor";
}
def InductorLayoutAttr : EnumAttr<Inductor_Dialect, InductorLayoutEnum, "layout">;

#endif // INDUCTOR_ATTRIBUTES
