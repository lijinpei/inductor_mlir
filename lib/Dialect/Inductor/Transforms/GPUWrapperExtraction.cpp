#include "inductor/Dialect/Inductor/IR/InductorOps.h"
#include "inductor/Dialect/Inductor/Transforms/Passes.h"

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

#include "fmt/format.h"

#include <cstddef>
#include <memory>
#include <sstream>

namespace mlir {
namespace inductor {
#define GEN_PASS_DEF_INDUCTORGPUWRAPPEREXTRACTION
#include "inductor/Dialect/Inductor/Transforms/Passes.h.inc"

namespace wrapper {

std::string WrapperSignature::toStr() const {
  std::stringstream ss;
  for (const auto &argDesc : args) {
    if (argDesc.kind == AK_Index) {
      ss << 'D';
    } else if (argDesc.kind == AK_Integer) {
      ss << 'I' << argDesc.integer.size_and_signed;
    } else {
      ss << 'M' << argDesc.memref.rank;
    }
  }
  return ss.str();
}

namespace {
template <typename T> void formatFlattenMemRef(T &buf, int64_t rank) {
  fmt::format_to(buf, R"(void*, void*, int64_t)");
  for (int64_t i = 0; i < rank; ++i) {
    fmt::format_to(buf, ", int64_t, int64_t");
  }
}

template <typename T>
void formatFlattenMemRefArgsDecl(T &buf, std::size_t idx, int64_t rank) {
  fmt::format_to(buf, "  auto arg{0}_sizes = arg{0}.sizes();\n", idx);
  fmt::format_to(buf, "  auto arg{0}_strides = arg{0}.strides();\n", idx);
  fmt::format_to(buf, "  int64_t arg{0}_dim = arg{0}.dim();\n", idx);
  fmt::format_to(buf, "  assert(arg{0}_dim == {1});\n", idx, rank);
  fmt::format_to(buf,
                 "  void* arg{0}_storage_ptr = "
                 "arg{0}.at::TensorBase::storage().mutable_data();\n",
                 idx);
  fmt::format_to(buf,
                 "  int64_t arg{0}_storage_offset = "
                 "arg{0}.at::TensorBase::storage_offset();\n",
                 idx);
  fmt::format_to(
      buf, "  void* arg{0}_data_ptr = arg{0}.at::TensorBase::data_ptr();\n",
      idx);
  for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
    fmt::format_to(buf, "  int64_t arg{0}_size{1} = arg{0}_sizes[{1}];\n", idx,
                   dimIdx);
    fmt::format_to(buf, "  int64_t arg{0}_stride{1} = arg{0}_strides[{1}];\n",
                   idx, dimIdx);
  }
}

template <typename T>
void formatFlattenMemRefArgsPassing(T &buf, std::size_t idx, int64_t rank) {
  fmt::format_to(
      buf, "arg{0}_storage_ptr, arg{0}_data_ptr, arg{0}_storage_offset", idx);
  for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
    fmt::format_to(buf, ", arg{0}_size{1}", idx, dimIdx);
  }
  for (int64_t dimIdx = 0; dimIdx < rank; ++dimIdx) {
    fmt::format_to(buf, ", arg{0}_stride{1}", idx, dimIdx);
  }
}

} // namespace

std::string generateWrapperDeclaration(const std::string &name,
                                       const WrapperSignature &signature) {
  auto out = fmt::memory_buffer();
  auto outIter = std::back_inserter(out);

  auto format_size_and_sign_as_type = [&](int64_t size_and_signed) {
    if (size_and_signed < 0) {
      fmt::format_to(outIter, "int{}_t", -size_and_signed);
    } else {
      fmt::format_to(outIter, "uint{}_t", size_and_signed);
    }
  };

  fmt::format_to(outIter,
                 R"=(
#ifdef ENABLE_PYTHNON_EXT

#define ENABLE_TORCH_CXX_EXT

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include "torch/csrc/autograd/python_variable.h"

#endif // ENABLE_PYTHNON_EXT

#ifdef __cplusplus
#include <cstdint>
#include <cassert>
#else
#include <stdint.h>
#include <assert.h>
#endif // __cplusplus

#ifdef ENABLE_TORCH_CXX_EXT
#include "ATen/ATen.h"
#endif

#ifdef __cplusplus
extern "C" {{
#endif

void {}()=",
                 name);
  const auto &args = signature.getArgs();
  for (const auto &[idx, arg] : llvm::enumerate(args)) {
    if (idx) {
      fmt::format_to(outIter, ", ");
    }
    switch (arg.kind) {
    case AK_Index:
      fmt::format_to(outIter, "int64_t");
      break;
    case AK_Integer:
      format_size_and_sign_as_type(arg.integer.size_and_signed);
      break;
    case AK_MemRef:
      formatFlattenMemRef(outIter, arg.memref.rank);
      break;
    default:
      llvm_unreachable("invalid argument kind");
    }
  }
  fmt::format_to(outIter, R"=();


#ifdef ENABLE_TORCH_CXX_EXT

void _torch_{}()=",
                 name);
  for (const auto &[idx, arg] : llvm::enumerate(args)) {
    if (idx) {
      fmt::format_to(outIter, ", ");
    }
    switch (arg.kind) {
    case AK_Index:
      fmt::format_to(outIter, "int64_t");
      break;
    case AK_Integer:
      format_size_and_sign_as_type(arg.integer.size_and_signed);
      break;
    case AK_MemRef:
      fmt::format_to(outIter, "const at::Tensor&");
      break;
    default:
      llvm_unreachable("invalid argument kind");
    }
    fmt::format_to(outIter, " arg{}", idx);
  }
  fmt::format_to(outIter, ") {{\n");

  for (const auto &[idx, arg] : llvm::enumerate(args)) {
    switch (arg.kind) {
    case AK_Index:
      break;
    case AK_Integer:
      break;
    case AK_MemRef:
      formatFlattenMemRefArgsDecl(outIter, idx, arg.memref.rank);
      break;
    default:
      llvm_unreachable("invalid argument kind");
    }
    fmt::format_to(outIter, "\n");
  }

  fmt::format_to(outIter, "  {}(", name);
  for (const auto &[idx, arg] : llvm::enumerate(args)) {
    if (idx) {
      fmt::format_to(outIter, ",    \n");
    }
    switch (arg.kind) {
    case AK_Index:
      fmt::format_to(outIter, "arg{}", idx);
      break;
    case AK_Integer:
      fmt::format_to(outIter, "arg{}", idx);
      break;
    case AK_MemRef:
      formatFlattenMemRefArgsPassing(outIter, idx, arg.memref.rank);
      break;
    default:
      llvm_unreachable("invalid argument kind");
    }
  }
  fmt::format_to(outIter, R"=();
}}

#endif // ENABLE_TORCH_CXX_EXT

#ifdef ENABLE_PYTHNON_EXT

static PyObject * _py_{}(PyObject *self,
                         PyObject *const *args,
                         Py_ssize_t nargs) {{
)=",
                 name);
  for (const auto &[idx, arg] : llvm::enumerate(args)) {
    switch (arg.kind) {
    case AK_Index:
      fmt::format_to(outIter, "  auto arg{0} = PyLong_AsLong(args[{0}]);\n",
                     idx);
      break;
    case AK_Integer:
      fmt::format_to(outIter, "  auto arg{0} = PyLong_AsLong(args[{0}]);\n",
                     idx);
      break;
    case AK_MemRef:
      fmt::format_to(outIter,
                     "  auto& arg{0} = THPVariable_Unpack(args[{0}]);\n", idx);
      break;
    default:
      llvm_unreachable("invalid argument kind");
    }
  }
  fmt::format_to(outIter, "  _torch_{}(", name);
  for (size_t idx = 0, e = args.size(); idx < e; ++idx) {
    if (idx) {
      fmt::format_to(outIter, ", ");
    }
    fmt::format_to(outIter, "arg{}", idx);
  }
  fmt::format_to(outIter, R"=();
  Py_INCREF(Py_None);
  return Py_None;
}}

static PyMethodDef _{0}_methods[] = {{
	{{ "call", _PyCFunction_CAST(_py_{0}), METH_FASTCALL,
	    "Call {0}." }},
	{{ NULL, NULL, 0, NULL }}
}};

static struct PyModuleDef _{0}_module = {{
	PyModuleDef_HEAD_INIT,
	"{0}",
	NULL,
	-1,
	_{0}_methods
}};

PyMODINIT_FUNC
PyInit_{0}(void) {{
  return PyModule_Create(&_{0}_module);
}}

#endif // ENABLE_PYTHNON_EXT

#ifdef __cplusplus
}} // extern "C"
#endif)=",
                 name);

  return to_string(out);
}

} // namespace wrapper
namespace {
struct GPUWrapperExtraction
    : public inductor::impl::InductorGPUWrapperExtractionBase<
          GPUWrapperExtraction> {
  using InductorGPUWrapperExtractionBase::InductorGPUWrapperExtractionBase;
  std::string *wrapper;
  GPUWrapperExtraction(std::string *wrapper = nullptr) : wrapper(wrapper) {}
  constexpr static const char *alloc_memref = "alloc_memref";
  FlatSymbolRefAttr getOrInsertMemRefAlloc(OpBuilder &builder,
                                           func::FuncOp funcOp,
                                           LLVMTypeConverter &typeConverter) {
    auto moduleOp = cast<ModuleOp>(funcOp->getParentOp());
    auto *context = builder.getContext();
    if (!moduleOp.lookupSymbol<LLVM::LLVMFuncOp>(alloc_memref)) {

      IRRewriter::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(moduleOp.getBody());
      Type indexType = typeConverter.getIndexType();
      Type ptrType = LLVM::LLVMPointerType::get(context);
      Type voidType = LLVM::LLVMVoidType::get(context);
      auto allocMemRefFuncType = LLVM::LLVMFunctionType::get(
          voidType, {indexType, ptrType, ptrType}, false);
      builder.create<LLVM::LLVMFuncOp>(moduleOp.getLoc(), alloc_memref,
                                       allocMemRefFuncType);
    }
    return SymbolRefAttr::get(context, alloc_memref);
  }

  Value convertAllocaToLibcall(func::FuncOp funcOp, memref::AllocaOp alloca,
                               OpBuilder &builder,
                               LLVMTypeConverter &typeConverter) {
    auto loc = alloca.getLoc();
    IRRewriter::InsertionGuard guard(builder);
    auto *context = builder.getContext();
    auto allocaVal = alloca.getResult();
    auto memrefType = cast<MemRefType>(allocaVal.getType());
    auto llvmMemRefType = typeConverter.convertType(memrefType);
    auto llvmIndexType = typeConverter.getIndexType();
    auto llvmPtrType = LLVM::LLVMPointerType::get(context);
    auto rank = memrefType.getRank();
    auto &entryBB = funcOp.getBody().front();
    builder.setInsertionPointToStart(&entryBB);
    auto c1 = builder.create<LLVM::ConstantOp>(loc, llvmIndexType, 1);
    Value memrefAlloca =
        builder.create<LLVM::AllocaOp>(loc, llvmPtrType, llvmMemRefType, c1);
    Value cRank = builder.create<LLVM::ConstantOp>(loc, llvmIndexType, rank);
    Value sizesAlloca =
        builder.create<LLVM::AllocaOp>(loc, llvmPtrType, llvmIndexType, cRank);
    builder.setInsertionPoint(alloca);
    for (int64_t i = 0, j = 0; i < rank; ++i) {
      auto size = memrefType.getDimSize(i);
      Value sizeVal;
      if (size == ShapedType::kDynamic) {
        sizeVal = builder
                      .create<UnrealizedConversionCastOp>(
                          loc, llvmIndexType, alloca.getDynamicSizes()[j++])
                      .getResult(0);
      } else {
        sizeVal = builder.create<LLVM::ConstantOp>(loc, llvmIndexType, size);
      }
      Value idx = builder.create<LLVM::ConstantOp>(loc, llvmIndexType, i);
      auto gep = builder.create<LLVM::GEPOp>(loc, llvmPtrType, llvmIndexType,
                                             sizesAlloca, idx);
      builder.create<LLVM::StoreOp>(loc, sizeVal, gep);
    }
    auto allocMemRefFunc =
        getOrInsertMemRefAlloc(builder, funcOp, typeConverter);
    SmallVector<Value> args = {cRank, memrefAlloca, sizesAlloca};
    builder.create<LLVM::CallOp>(loc, TypeRange{}, allocMemRefFunc, args);
    Value llvmRes =
        builder.create<LLVM::LoadOp>(loc, llvmMemRefType, memrefAlloca);
    return builder.create<UnrealizedConversionCastOp>(loc, memrefType, llvmRes)
        .getResult(0);
  }

  void recordWrapperInfoOnModule(const wrapper::WrapperInfo &wrapperInfo,
                                 func::FuncOp funcOp, OpBuilder &builder) {
    IRRewriter::InsertionGuard guard(builder);
    builder.setInsertionPoint(funcOp);
    auto sigStr = wrapperInfo.signature.toStr();
    auto loc = funcOp.getLoc();
    auto memrefType = MemRefType::get(sigStr.size() + 1, builder.getI8Type());
    auto tensorType =
        RankedTensorType::get(sigStr.size() + 1, builder.getI8Type());
    builder.create<memref::GlobalOp>(
        loc, "__sig_" + wrapperInfo.name, builder.getStringAttr("public"),
        memrefType,
        DenseElementsAttr::getFromRawBuffer(
            tensorType, ArrayRef<char>(sigStr.data(), sigStr.size() + 1)),
        true, builder.getI64IntegerAttr(1));
  }
  wrapper::WrapperInfo generateWrapperInfo(func::FuncOp funcOp) {
    wrapper::WrapperInfo wrapper;
    wrapper.name = funcOp.getSymName().str();
    assert(funcOp.getResultTypes().size() == 0);
    for (auto argTy : funcOp.getArgumentTypes()) {
      wrapper::ArgDesc argDesc;
      if (argTy.isIndex()) {
        argDesc.kind = wrapper::AK_Index;
      } else if (argTy.isInteger()) {
        bool isSigned = argTy.isSignedInteger();
        int64_t size = argTy.getIntOrFloatBitWidth();
        argDesc.kind = wrapper::AK_Integer;
        argDesc.integer.size_and_signed = isSigned ? -size : size;
      } else {
        argTy.dump();
        auto memRef = dyn_cast<MemRefType>(argTy);
        assert(memRef);
        argDesc.kind = wrapper::AK_MemRef;
        argDesc.memref.rank = memRef.getRank();
      }
      wrapper.signature.addArg(argDesc);
    }
    return wrapper;
  }

  void extractWrapperForFunc(func::FuncOp funcOp) {
    SmallVector<Type> addedArgTys;
    SmallVector<memref::AllocaOp> allocas;
    SmallVector<memref::AllocOp> allocs;
    funcOp.walk([&](Operation *op) {
      if (auto allocaOp = dyn_cast<memref::AllocaOp>(op)) {
        allocas.push_back(allocaOp);
        addedArgTys.push_back(allocaOp.getResult().getType());
      } else if (auto allocOp = dyn_cast<memref::AllocOp>(op)) {
        allocs.push_back(allocOp);
      }
    });
    auto launcherType = funcOp.getFunctionType();
    SmallVector<Type> kernelArgTys;
    auto num_input_args = launcherType.getNumInputs();
    auto num_output_args = launcherType.getNumResults();
    auto num_added_args = addedArgTys.size();
    kernelArgTys.reserve(num_output_args + num_input_args + num_added_args);
    auto resultTys = launcherType.getResults();
    kernelArgTys.append(resultTys.begin(), resultTys.end());
    auto inputTys = launcherType.getInputs();
    kernelArgTys.append(inputTys.begin(), inputTys.end());
    kernelArgTys.append(addedArgTys);
    OpBuilder builder(funcOp);
    auto kernelFuncTy = builder.getFunctionType(kernelArgTys, {});
    auto loc = funcOp.getLoc();
    auto kernelFunc =
        builder.create<func::FuncOp>(loc, "device_kernel", kernelFuncTy);
    IRMapping mapping;
    auto &kernelEntryBB = *kernelFunc.addEntryBlock();
    auto &launcherEntryBB = funcOp.getBody().front();
    for (size_t i = 0; i < num_input_args; ++i) {
      mapping.map(launcherEntryBB.getArgument(i),
                  kernelEntryBB.getArgument(i + num_output_args));
    }
    builder.cloneRegionBefore(funcOp.getBody(), kernelFunc.getBody(),
                              kernelFunc.getBody().end(), mapping);
    auto &kernelSecondBB = *kernelEntryBB.getNextNode();
    kernelEntryBB.getOperations().splice(kernelEntryBB.end(),
                                         kernelSecondBB.getOperations());
    kernelSecondBB.erase();
    for (size_t i = 0; i < num_output_args; ++i) {
      auto oldAlloc = allocs[i];
      auto newAlloc =
          cast<memref::AllocOp>(mapping.lookup(oldAlloc.getOperation()));
      newAlloc.getResult().replaceAllUsesWith(
          Value(kernelEntryBB.getArgument(i)));
      newAlloc->erase();
    }
    for (size_t i = 0; i < num_added_args; ++i) {
      auto oldAlloca = allocas[i];
      auto newAlloca =
          cast<memref::AllocaOp>(mapping.lookup(oldAlloca.getOperation()));
      newAlloca.getResult().replaceAllUsesWith(Value(
          kernelEntryBB.getArgument(num_output_args + num_input_args + i)));
      newAlloca->erase();
    }
    funcOp.walk([&](Operation *op) {
      if (isa<inductor::PointwiseLoopOp, inductor::ReductionLoopOp>(op)) {
        op->erase();
      }
    });
    kernelFunc.walk([&](func::ReturnOp retOp) {
      builder.setInsertionPoint(retOp);
      builder.create<func::ReturnOp>(retOp.getLoc());
      retOp.erase();
    });
    SmallVector<Value> kernelArgs;
    kernelArgs.append(allocs.begin(), allocs.end());
    kernelArgs.append(launcherEntryBB.args_begin(), launcherEntryBB.args_end());
    kernelArgs.append(allocas.begin(), allocas.end());
    // auto &hostEntryBB = funcOp.getBody().front();
    // LLVMTypeConverter typeConverter(builder.getContext());
    // for (auto alloca : allocas) {
    //   auto val = convertAllocaToLibcall(funcOp, alloca, builder,
    //   typeConverter); alloca.getResult().replaceAllUsesWith(val);
    //   kernelArgs.push_back(val);
    // }
    builder.setInsertionPoint(&launcherEntryBB,
                              std::prev(launcherEntryBB.end()));
    builder.create<func::CallOp>(loc, kernelFunc, kernelArgs);
    return;
    auto wrapperInfo = generateWrapperInfo(funcOp);
    recordWrapperInfoOnModule(wrapperInfo, funcOp, builder);
    if (wrapper) {
      *wrapper =
          generateWrapperDeclaration(wrapperInfo.name, wrapperInfo.signature);
      llvm::errs() << "wrapper code:\n";
      llvm::errs() << wrapper << '\n';
    }
  }

  void runOnOperation() override {
    auto modOp = cast<ModuleOp>(getOperation());
    for (auto &op : modOp.getBodyRegion().front()) {
      extractWrapperForFunc(cast<func::FuncOp>(&op));
    }
  }
};
} // namespace
std::unique_ptr<Pass>
createInductorGPUWrapperExtractionPass(std::string *wrapper) {
  return std::make_unique<GPUWrapperExtraction>(wrapper);
}
} // namespace inductor
} // namespace mlir
