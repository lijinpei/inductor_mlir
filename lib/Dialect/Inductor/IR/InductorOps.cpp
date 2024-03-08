#include "inductor/Dialect/Inductor/IR/InductorOps.h"

#include "inductor/Dialect/Inductor/IR/InductorAttrs.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"

#define GET_OP_CLASSES

#include "inductor/Dialect/Inductor/IR/Inductor.cpp.inc"

using namespace mlir;

void inductor::ReductionOp::build(
    OpBuilder &builder, OperationState &odsState,
    ::mlir::inductor::DeviceAttr device, ::mlir::ValueRange ranges,
    ::mlir::ValueRange reduction_ranges, ::mlir::StringAttr reduction_type,
    ::mlir::inductor::ReductionHintAttr reduction_hint) {
  OpBuilder::InsertionGuard guard(builder);

  odsState.addOperands(ranges);
  odsState.addOperands(reduction_ranges);
  ::llvm::copy(
      ::llvm::ArrayRef<int32_t>(
          {static_cast<int32_t>(ranges.size()),
           static_cast<int32_t>(reduction_ranges.size())}),
      odsState.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  odsState.getOrAddProperties<Properties>().device = device;
  odsState.getOrAddProperties<Properties>().reduction_type = reduction_type;
  odsState.getOrAddProperties<Properties>().reduction_hint = reduction_hint;

  Region *innerFnRegion = odsState.addRegion();
  Block *bodyBlock = builder.createBlock(innerFnRegion);
  for (Value val : ranges)
    bodyBlock->addArgument(val.getType(), val.getLoc());
  for (Value val : reduction_ranges)
    bodyBlock->addArgument(val.getType(), val.getLoc());
}

void inductor::ComputedBufferOp::build(::mlir::OpBuilder &builder,
                                       ::mlir::OperationState &odsState,
                                       /*optional*/ ::mlir::StringAttr name) {
  OpBuilder::InsertionGuard guard(builder);
  if (name) {
    odsState.getOrAddProperties<Properties>().name = name;
  }
  Region *region = odsState.addRegion();
  builder.createBlock(region);
}

inductor::LoopsOpInterface inductor::ComputedBufferOp::getLoop() {
  for (auto &op : getDataRegion().front()) {
    if (auto loopOp = dyn_cast<inductor::LoopsOpInterface>(&op)) {
      return loopOp;
    }
  }
  return {};
}

void inductor::PlainStorageOp::build(
    ::mlir::OpBuilder &builder, ::mlir::OperationState &state,
    ::mlir::Type result, ::mlir::StringAttr name,
    ::mlir::inductor::DeviceAttr device, ::mlir::TypeAttr dtype,
    ::mlir::ValueRange size, ::mlir::ValueRange stride, ::mlir::Value offset,
    /*optional*/ ::mlir::UnitAttr is_fixed) {
  OpBuilder::InsertionGuard guard(builder);

  state.addOperands(size);
  state.addOperands(stride);
  state.addOperands(offset);
  ::llvm::copy(
      ::llvm::ArrayRef<int32_t>({static_cast<int32_t>(size.size()),
                                 static_cast<int32_t>(stride.size()), 1}),
      state.getOrAddProperties<Properties>().operandSegmentSizes.begin());
  state.getOrAddProperties<Properties>().name = name;
  state.getOrAddProperties<Properties>().device = device;
  state.getOrAddProperties<Properties>().dtype = dtype;
  if (is_fixed) {
    state.getOrAddProperties<Properties>().is_fixed = is_fixed;
  }
  state.addTypes(result);
  Region *region = state.addRegion();
  builder.createBlock(region);
}
