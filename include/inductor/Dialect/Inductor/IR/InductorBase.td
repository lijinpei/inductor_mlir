#ifndef INDUCTOR_BASE
#define INDUCTOR_BASE

include "mlir/IR/DialectBase.td"

def Inductor_Dialect : Dialect {
	let name = "inductor";
	let cppNamespace = "::mlir::inductor";
	let dependentDialects = [
		"affine::AffineDialect",
		"arith::ArithDialect",
		"index::IndexDialect",
		"math::MathDialect",
		"memref::MemRefDialect",
	];
	let useDefaultAttributePrinterParser = 1;
}

#endif // INDUCTOR_BASE
