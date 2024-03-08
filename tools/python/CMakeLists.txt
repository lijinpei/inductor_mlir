get_property(PYTHON_WRAPPER_SRCS GLOBAL PROPERTY PYTHON_WRAPPER_SRCS)
get_property(TRITON_LIBS GLOBAL PROPERTY TRITON_LIBS)
pybind11_add_module(inductor_mlir inductor_mlir.cpp ir_builder.cpp passes.cpp ${PYTHON_WRAPPER_SRCS})
target_link_libraries(inductor_mlir PRIVATE InductorIR InductorToTriton ${TRITON_LIBS})
