#include "inductor/Conversion/InductorToTriton/Passes.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/Debug.h"

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

#include <string>
#include <vector>

namespace py = pybind11;
using ret = py::return_value_policy;
using namespace mlir;

void init_passes(py::module &&m) {
  py::class_<PassManager>(m, "PassManager")
      .def(py::init<MLIRContext *>())
      .def("run", [](PassManager &self, OpState &mod) {
        // FIXME: setup python exception?
        if (failed(self.run(mod.getOperation())))
          throw std::runtime_error("PassManager::run failed");
      });
  m.def("add_convert_inductor_to_triton", [](mlir::PassManager &pm) {
    pm.addPass(inductor::createConvertInductorToTritonPass());
  });
  m.def("enable_debug", [](bool on) { llvm::DebugFlag = on; });
  m.def("debug_only", [](const std::string &str) {
    llvm::DebugFlag = true;
    llvm::setCurrentDebugType(str.data());
  });
  m.def("debug_only", [](const std::vector<std::string> strs) {
    llvm::DebugFlag = true;
    std::vector<const char *> ptrs;
    ptrs.reserve(strs.size());
    for (const auto &str : strs) {
      ptrs.push_back(str.data());
    }
    llvm::setCurrentDebugTypes(ptrs.data(), ptrs.size());
  });
}
