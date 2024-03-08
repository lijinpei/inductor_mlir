#pragma once

#include "mlir/Pass/Pass.h"

#include <memory>

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace inductor {

std::unique_ptr<OperationPass<ModuleOp>> createConvertInductorToTritonPass();

#define GEN_PASS_DECL
#include "inductor/Conversion/InductorToTriton/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "inductor/Conversion/InductorToTriton/Passes.h.inc"

} // namespace inductor

} // namespace mlir
