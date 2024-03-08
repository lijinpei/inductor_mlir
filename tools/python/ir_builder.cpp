#include "ir_builder.h"

#include "inductor/Dialect/Inductor/IR/InductorAttrs.h"
#include "inductor/Dialect/Inductor/IR/InductorDialect.h"
#include "inductor/Dialect/Inductor/IR/InductorOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"

#include "llvm/Support/ErrorHandling.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;
using ret = py::return_value_policy;
using namespace mlir;

InductorIRBuilder::InductorIRBuilder(MLIRContext &context) : context(context) {
  context.loadDialect<mlir::inductor::InductorDialect>();
}

int64_t InductorIRBuilder::getKDynamic() { return ShapedType::kDynamic; }

OpBuilder::InsertPoint InductorIRBuilder::getInsertPoint() {
  return builder.saveInsertionPoint();
}

void InductorIRBuilder::setInsertPoint(OpBuilder::InsertPoint point) {
  return builder.restoreInsertionPoint(point);
}

Location InductorIRBuilder::getFileLineColLoc(const std::string &filename,
                                              unsigned line, unsigned column) {
  return FileLineColLoc::get(&context, filename, line, column);
}

Location InductorIRBuilder::getUnknownLoc() {
  return UnknownLoc::get(&context);
}

Attribute InductorIRBuilder::getReductionHintAttr(int64_t val) {
  assert(val >= (int64_t)inductor::ReductionHint::INNER &&
         val <= (int64_t)inductor::ReductionHint::DEFAULT);
  auto *context = getContext();
  return inductor::ReductionHintAttr::get(context,
                                          (inductor::ReductionHint)val);
}

Type InductorIRBuilder::getF32Type() { return builder.getF32Type(); }

Type InductorIRBuilder::getMemRefType(const std::vector<int64_t> &shape,
                                      Type elemType) {
  return MemRefType::get(shape, elemType);
}

OpState InductorIRBuilder::createModuleOp(Location loc,
                                          std::optional<std::string> name_) {
  std::optional<StringRef> name;
  if (name_) {
    name = *name_;
  }
  auto res = ModuleOp::create(loc, name);
  builder.setInsertionPointToStart(&res.getBodyRegion().front());
  top_level_modules.push_back(res);
  return res;
}

void InductorIRBuilder::dump() {
  for (auto mod : top_level_modules) {
    mod.dump();
  }
}

void init_ir_builder(py::module &&m) {
  py::class_<MLIRContext>(m, "MLIRContext").def(py::init<>());
  py::class_<mlir::Location>(m, "Location");
  py::class_<mlir::OpState>(m, "Operation")
      .def("result", [](OpState &self, unsigned idx) -> Value {
        if (idx >= self->getNumResults())
          throw pybind11::index_error("Op result index out of range");
        return self->getResult(idx);
      });
  py::class_<mlir::Type>(m, "Type");
  py::class_<mlir::Attribute>(m, "Attribute");
  py::class_<mlir::Value>(m, "Value");
  py::class_<mlir::OpBuilder::InsertPoint>(m, "InsertPoint");

  // Copied from triton/python/src/ir.cc
  py::class_<Region>(m, "region", py::module_local())
      .def("front", &Region::front, ret::reference)
      .def("back", &Region::back, ret::reference);
  py::class_<BlockArgument, Value>(m, "block_argument", py::module_local());
  py::class_<Block>(m, "block", py::module_local())
      .def("arg",
           [](Block &self, unsigned index) -> BlockArgument {
             if (index >= self.getNumArguments())
               throw pybind11::index_error("Block argument index out of range");
             return self.getArgument(index);
           })
      .def("get_num_arguments", &Block::getNumArguments);

  py::class_<InductorIRBuilder>(m, "IRBuilder")
      .def(py::init<MLIRContext &>())
      .def("empty_value",
           [](InductorIRBuilder &self) -> Value { return Value(); })
      .def("set_insertion_point_to_start",
           [](InductorIRBuilder &self, Block &block) -> void {
             self.getBuilder().setInsertionPointToStart(&block);
           })
      .def("set_insertion_point_to_end",
           [](InductorIRBuilder &self, Block &block) {
             self.getBuilder().setInsertionPointToEnd(&block);
           })
      .def("get_insertion_point",
           [](InductorIRBuilder &self) {
             return self.getBuilder().saveInsertionPoint();
           })
      .def("set_insertion_point",
           [](InductorIRBuilder &self, OpBuilder::InsertPoint pt) {
             self.getBuilder().restoreInsertionPoint(pt);
           })
      .def("get_kdynamic", &InductorIRBuilder::getKDynamic)
      .def("get_file_line_col_loc", &InductorIRBuilder::getFileLineColLoc)
      .def("create_unknown_loc", &InductorIRBuilder::getUnknownLoc)
      .def("get_f32_type", &InductorIRBuilder::getF32Type)
      .def("get_memref_type", &InductorIRBuilder::getMemRefType)
      .def("get_reduction_hint_attr", &InductorIRBuilder::getReductionHintAttr)
      .def("create_module_op", &InductorIRBuilder::createModuleOp)
      .def("create_calc_plain_offset_op",
           [](InductorIRBuilder &self, Location loc,
              const std::vector<Value> &coord, const std::vector<Value> &size,
              const std::vector<Value> &stride, Value offset) -> OpState {
             auto &builder = self.getBuilder();
             auto i32Ty = builder.getI32Type();
             return builder.create<inductor::CalcPlainOffsetOp>(
                 loc, i32Ty, coord, size, stride, offset);
           })
      .def("create_constant_i32_op",
           [](InductorIRBuilder &self, Location loc, int64_t val) -> OpState {
             auto &builder = self.getBuilder();
             auto i32Ty = builder.getI32Type();
             auto valAttr = builder.getIntegerAttr(i32Ty, val);
             return builder.create<arith::ConstantOp>(loc, valAttr);
           })
      .def("create_load_f32_op",
           [](InductorIRBuilder &self, Location loc, const std::string &buffer,
              Value offset) -> OpState {
             // FIXME: non fp32 load
             auto &builder = self.getBuilder();
             auto f32Ty = builder.getF32Type();
             auto bufferStr = builder.getStringAttr(buffer);
             return builder.create<inductor::LoadOp>(loc, f32Ty, bufferStr,
                                                     offset);
           })
      .def("create_yield_op",
           [](InductorIRBuilder &self, Location loc, Value value) -> OpState {
             auto &builder = self.getBuilder();
             return builder.create<inductor::YieldOp>(loc, value);
           })
      .def("create_constant_f32_op",
           [](InductorIRBuilder &self, Location loc, float val) -> OpState {
             auto &builder = self.getBuilder();
             auto valAttr = builder.getF32FloatAttr(val);
             return builder.create<arith::ConstantOp>(loc, valAttr);
           })
      .def("create_sin_op",
           [](InductorIRBuilder &self, Location loc, Value val) -> OpState {
             auto &builder = self.getBuilder();
             auto *context = self.getContext();
             // FIXME: fast-math flags
             auto fmf = arith::FastMathFlagsAttr::get(
                 context, arith::FastMathFlags::none);
             return builder.create<math::SinOp>(loc, val, fmf);
           })
      .def("create_cos_op",
           [](InductorIRBuilder &self, Location loc, Value val) -> OpState {
             auto &builder = self.getBuilder();
             auto *context = self.getContext();
             // FIXME: fast-math flags
             auto fmf = arith::FastMathFlagsAttr::get(
                 context, arith::FastMathFlags::none);
             return builder.create<math::CosOp>(loc, val, fmf);
           })
      .def("create_addf_op",
           [](InductorIRBuilder &self, Location loc, Value val1,
              Value val2) -> OpState {
             auto &builder = self.getBuilder();
             auto *context = self.getContext();
             // FIXME: fast-math flags
             auto fmf = arith::FastMathFlagsAttr::get(
                 context, arith::FastMathFlags::none);
             return builder.create<arith::AddFOp>(loc, val1, val2, fmf);
           })
      .def("create_storage_op_from_layout",
           [](InductorIRBuilder &self, Location loc,
              const std::string &layoutKind, const std::string &device,
              Type dtype, const std::vector<int64_t> &staticSize,
              const std::vector<Value> &dynamicSize,
              const std::vector<Value> &stride, Value offset,
              Value viewOrTarget, const std::string &name) -> OpState {
             auto &builder = self.getBuilder();
             auto *context = self.getContext();
             bool isFixedLayout = layoutKind == "FixedLayout";
             bool isFlexibleLayout = layoutKind == "FlexibleLayout";
             if (isFixedLayout || isFlexibleLayout) {
               assert(!viewOrTarget);
               UnitAttr isFixedAttr;
               if (isFixedLayout) {
                 isFixedAttr = builder.getUnitAttr();
               }
               auto nameStr = builder.getStringAttr(name);
               auto devStr = builder.getStringAttr(device);
               auto devAttr = inductor::DeviceAttr::get(context, devStr);
               auto dtypeAttr = TypeAttr::get(dtype);
               auto resultType = MemRefType::get(staticSize, dtype);
               return builder.create<inductor::PlainStorageOp>(
                   loc, resultType, nameStr, devAttr, dtypeAttr, dynamicSize,
                   stride, offset, isFixedAttr);
             } else {
               assert(false);
             }
             llvm_unreachable("");
           })
      .def("create_input_buffer_op",
           [](InductorIRBuilder &self, Location loc,
              const std::optional<std::string> &name) -> OpState {
             auto &builder = self.getBuilder();
             StringAttr nameAttr;
             if (name) {
               nameAttr = builder.getStringAttr(*name);
             }
             return builder.create<inductor::InputBufferOp>(loc, nameAttr);
           })
      .def("create_computed_buffer_op",
           [](InductorIRBuilder &self, Location loc,
              const std::optional<std::string> &name) -> OpState {
             auto &builder = self.getBuilder();
             StringAttr nameAttr;
             if (name) {
               nameAttr = builder.getStringAttr(*name);
             }
             return builder.create<inductor::ComputedBufferOp>(loc, nameAttr);
           })
      .def("create_reduction_op",
           [](InductorIRBuilder &self, Location loc, const std::string &device,
              const std::vector<Value> &ranges,
              const std::vector<Value> &reduction_ranges,
              const std::string &reduction_type,
              int64_t reduction_hint) -> OpState {
             auto &builder = self.getBuilder();
             auto *context = self.getContext();
             auto devStr = builder.getStringAttr(device);
             auto devAttr = inductor::DeviceAttr::get(context, devStr);
             auto redTyAttr = builder.getStringAttr(reduction_type);
             auto redHintAttr = cast<inductor::ReductionHintAttr>(
                 self.getReductionHintAttr(reduction_hint));
             return builder.create<inductor::ReductionOp>(
                 loc, devAttr, ranges, reduction_ranges, redTyAttr,
                 redHintAttr);
           })
      .def(
          "get_entry_bb_of_region",
          [](InductorIRBuilder &self, OpState &op_,
             unsigned regionIdx) -> Block & {
            auto *op = op_.getOperation();
            return op->getRegion(regionIdx).front();
          },
          ret::reference)
      .def(
          "get_reduction_op_inner_fn",
          [](InductorIRBuilder &self, OpState &op_) -> Region & {
            auto *op = op_.getOperation();
            auto redOp = cast<inductor::ReductionOp>(op);
            return redOp.getInnerFn();
          },
          ret::reference)
      .def("dump", &InductorIRBuilder::dump);
}
