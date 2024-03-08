#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace inductor {
namespace impl {
Operation *getDataOpFromDataRegion(Region &dataRegion);
} // namespace impl
} // namespace inductor
} // namespace mlir

#include "inductor/Dialect/Inductor/IR/InductorInterfaces.h.inc"
