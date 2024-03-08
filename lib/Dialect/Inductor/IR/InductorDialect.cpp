#include "inductor/Dialect/Inductor/IR/InductorDialect.h"
#include "inductor/Dialect/Inductor/IR/InductorAttrs.h"
#include "inductor/Dialect/Inductor/IR/InductorOps.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/TypeSwitch.h"

#include "inductor/Dialect/Inductor/IR/InductorDialect.cpp.inc"

using namespace mlir;

void inductor::InductorDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "inductor/Dialect/Inductor/IR/InductorAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "inductor/Dialect/Inductor/IR/Inductor.cpp.inc"
      >();
}

#include "inductor/Dialect/Inductor/IR/InductorEnums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "inductor/Dialect/Inductor/IR/InductorAttrs.cpp.inc"
