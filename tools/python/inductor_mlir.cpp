#include <pybind11/pybind11.h>

namespace py = pybind11;

void init_ir_builder(py::module &&m);
void init_passes(py::module &&m);

PYBIND11_MODULE(inductor_mlir, m) {
  m.doc() = "Python bindings for inductor MLIR dialect";
  init_ir_builder(m.def_submodule("ir_builder"));
  init_passes(m.def_submodule("passes"));
}
