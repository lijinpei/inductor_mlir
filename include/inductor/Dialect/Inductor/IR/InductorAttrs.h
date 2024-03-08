#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "inductor/Dialect/Inductor/IR/InductorEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "inductor/Dialect/Inductor/IR/InductorAttrs.h.inc"
