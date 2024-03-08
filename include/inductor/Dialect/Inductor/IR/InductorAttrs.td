#ifndef INDUCTOR_ATTRIBUTES
#define INDUCTOR_ATTRIBUTES

include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributes.td"

include "inductor/Dialect/Inductor/IR/InductorBase.td"

class Inductor_Attr<string name, string mnem>
	: AttrDef<Inductor_Dialect, name> {
	let mnemonic = mnem;
}


def InductorReductionTypeEnum : I32EnumAttr<"ReductionType", "", [
I32EnumAttrCase<"ANY", 0>,
I32EnumAttrCase<"MAX", 1>,
I32EnumAttrCase<"MIN", 2>,
I32EnumAttrCase<"PROD", 3>,
I32EnumAttrCase<"SUM", 4>,
I32EnumAttrCase<"XOR_SUM", 5>,
I32EnumAttrCase<"ARGMIN", 6>,
I32EnumAttrCase<"ARGMAX", 7>,
I32EnumAttrCase<"WELFORD_COMBINE", 8>,
]> {
	let genSpecializedAttr = 0;
	let cppNamespace = "::mlir::inductor";
}
def InductorReductionTypeAttr : EnumAttr<Inductor_Dialect, InductorReductionTypeEnum, "reduction_type">;

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

#endif // INDUCTOR_ATTRIBUTES
