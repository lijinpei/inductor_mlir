#include "inductor/Conversion/InductorToAffine/Passes.h"
#include "inductor/Dialect/Inductor/IR/InductorDialect.h"
#include "inductor/Dialect/Inductor/Transforms/Passes.h"

#include "mlir-c/IR.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"
#include "mlir/CAPI/IR.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "torch/extension.h"

#include "pybind11/pybind11.h"

#include <memory>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;

namespace py = pybind11;

// -----------------------------------------------------------------------------
// Module initialization.
// -----------------------------------------------------------------------------

namespace {

std::unique_ptr<llvm::TargetMachine> createTargetMachine() {
  llvm::Triple triple = llvm::Triple(llvm::sys::getProcessTriple());
  llvm::SubtargetFeatures features;
  llvm::StringMap<bool> featureMap;
  llvm::sys::getHostCPUFeatures(featureMap);
  for (auto &feature : featureMap)
    features.AddFeature(feature.first(), feature.second);
  auto cpu = llvm::sys::getHostCPUName();
  std::string ErrMsg;
  auto *target = llvm::TargetRegistry::lookupTarget(triple.getTriple(), ErrMsg);
  if (!target) {
    llvm::errs() << ErrMsg << '\n';
    return {};
  }
  llvm::TargetOptions options;
  auto *tm = target->createTargetMachine(
      triple.getTriple(), cpu, features.getString(), options, llvm::Reloc::PIC_,
      /* CodeModel */ std::nullopt, llvm::CodeGenOptLevel::Default,
      /*JIT*/ false);
  if (!tm) {
    llvm::errs() << "failed to create target machine\n";
  }
  return std::unique_ptr<llvm::TargetMachine>(tm);
}

void runJit(MLIRContext &context, mlir::ModuleOp module) {
  // Initialize LLVM targets.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  // Register the translation from MLIR to LLVM IR, which must happen before we
  // can JIT-compile.
  mlir::registerBuiltinDialectTranslation(context);
  mlir::registerLLVMDialectTranslation(context);

  // An optimization pipeline to use within the execution engine.
  auto optPipeline = mlir::makeOptimizingTransformer(0, /*sizeLevel=*/0,
                                                     /*targetMachine=*/nullptr);

  // Create an MLIR execution engine. The execution engine eagerly JIT-compiles
  // the module.
  mlir::ExecutionEngineOptions engineOptions;
  engineOptions.transformer = optPipeline;
  SmallVector<StringRef> libPaths;
  libPaths.push_back("/home/lijinpei/libruntime.so");
  engineOptions.sharedLibPaths = libPaths;
  engineOptions.enableObjectDump = true;
  auto maybeEngine = mlir::ExecutionEngine::create(module, engineOptions);
  if (!maybeEngine) {
    llvm::errs() << "failed to construct an execution engine";
    return;
  }
  auto &engine = maybeEngine.get();
  engine->dumpToObjectFile("/home/lijinpei/1.obj");
  SmallVector<void *> args;
  int64_t size = 1000;
  int64_t c0 = 0;
  int64_t c1 = 1;
  void *out_data = malloc(sizeof(float) * size);
  void *in_data0 = malloc(sizeof(float) * size * size);
  void *in_data1 = malloc(sizeof(float) * size * size);
  args.push_back(&out_data);
  args.push_back(&out_data);
  args.push_back(&c0);
  args.push_back(&size);
  args.push_back(&c1);
  args.push_back(&size);
  args.push_back(&in_data0);
  args.push_back(&in_data0);
  args.push_back(&c0);
  args.push_back(&size);
  args.push_back(&size);
  args.push_back(&size);
  args.push_back(&c1);
  args.push_back(&in_data1);
  args.push_back(&in_data1);
  args.push_back(&c0);
  args.push_back(&size);
  args.push_back(&size);
  args.push_back(&size);
  args.push_back(&c1);

  // Invoke the JIT-compiled function.
  auto invocationResult = engine->invokePacked("my_kernel", args);
  if (invocationResult) {
    llvm::errs() << "JIT invocation failed\n";
  }
}
void compileModule(MLIRContext &context, ModuleOp mod, std::string &asmContent,
                   std::string &wrapperContent) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  context.disableMultithreading();
  PassManager PM(&context, ModuleOp::getOperationName());
  PM.enableVerifier(true);
  PM.enableIRPrinting([](Pass *, Operation *) { return true; },
                      [](Pass *, Operation *) { return true; }, true, false);
  PM.addPass(createPrintOpStatsPass());
  PM.addPass(inductor::createInductorGPUWrapperExtractionPass(&wrapperContent));
  PM.addPass(inductor::createConvertInductorToAffinePass());
  PM.addPass(createLowerAffinePass());
  PM.addPass(createCanonicalizerPass());
  PM.addPass(createCSEPass());
  PM.addPass(createConvertSCFToCFPass());
  // PM.addPass(createRemoveDeadValuesPass());

  PM.addPass(createFinalizeMemRefToLLVMConversionPass());
  PM.addPass(createConvertFuncToLLVMPass());
  PM.addPass(createConvertIndexToLLVMPass());
  PM.addNestedPass<func::FuncOp>(createConvertMathToLLVMPass());
  PM.addNestedPass<LLVM::LLVMFuncOp>(createConvertMathToLLVMPass());
  PM.addPass(createCanonicalizerPass());
  PM.addPass(createCSEPass());
  PM.addPass(createReconcileUnrealizedCastsPass());

  if (PM.run(mod).succeeded()) {
    llvm::errs() << "compile-success\n";
  } else {
    llvm::errs() << "compile-failed\n";
  }

  std::unique_ptr<llvm::LLVMContext> llvmContext(new llvm::LLVMContext);
  auto llvmModule = translateModuleToLLVMIR(mod, *llvmContext);
  if (!llvmModule) {
    llvm::errs() << "translate-to-llvm-failed\n";
  } else {
    llvm::errs() << "translate-to-llvm-success\n";
    llvmModule->dump();
  }
  auto tm = createTargetMachine();
  if (!tm) {
    llvm::errs() << "failed to create target machine\n";
    return;
  }
  llvmModule->setDataLayout(tm->createDataLayout());
  llvmModule->setTargetTriple(tm->getTargetTriple().getTriple());

  // TODO: optimize llvm ir
  llvm::raw_string_ostream stream(asmContent);

  { // Drop pstream after this to prevent the ISA from being stuck buffering
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;

    if (tm->addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "failed to add pass to emit file\n";
    }
    codegenPasses.run(*llvmModule);
  }
  return;
}

torch::Tensor d_tanh(torch::Tensor z) { return 1 - z.tanh().pow(2); }

} // namespace

PYBIND11_MODULE(_mlirDialectsInductor, m) {
  m.doc() = "MLIR Inductor Dialect";
  m.def("register_inductor_dialect", [](MlirDialectRegistry registry) {
    unwrap(registry)->insert<inductor::InductorDialect>();
  });
  m.def("compile_module",
        [](MlirContext context,
           MlirModule mod) -> std::tuple<std::string, std::string> {
          std::string asmContent, wrapperContent;
          compileModule(*unwrap(context), cast<ModuleOp>(unwrap(mod)),
                        asmContent, wrapperContent);
          return {asmContent, wrapperContent};
        });
  m.def("run_jit", [](MlirContext context, MlirModule mod) {
    return runJit(*unwrap(context), cast<ModuleOp>(unwrap(mod)));
  });
  m.def("d_tanh", &d_tanh);
}
