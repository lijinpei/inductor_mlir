#pragma once

#include "inductor/Dialect/Inductor/IR/InductorAttrs.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "inductor/Dialect/Inductor/IR/InductorInterfaces.h"

#define GET_OP_CLASSES
#include "inductor/Dialect/Inductor/IR/Inductor.h.inc"
