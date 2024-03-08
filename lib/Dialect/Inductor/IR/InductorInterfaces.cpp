#include "inductor/Dialect/Inductor/IR/InductorInterfaces.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Region.h"

#include "inductor/Dialect/Inductor/IR/InductorInterfaces.cpp.inc"

namespace mlir {
namespace inductor {
namespace impl {
Operation *getDataOpFromDataRegion(Region &dataRegion) {
  for (auto &subOp : dataRegion.front()) {
    if (isa<BufferOpInterface, LoopsOpInterface>(&subOp)) {
      return &subOp;
    }
  }
  return nullptr;
}
} // namespace impl
} // namespace inductor
} // namespace mlir
