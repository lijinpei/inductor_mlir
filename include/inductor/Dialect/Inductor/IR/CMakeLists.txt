add_mlir_dialect(Inductor inductor)

# Add Inductor attributes
set(LLVM_TARGET_DEFINITIONS InductorAttrs.td)
mlir_tablegen(InductorEnums.h.inc -gen-enum-decls)
mlir_tablegen(InductorEnums.cpp.inc -gen-enum-defs)
mlir_tablegen(InductorAttrs.h.inc -gen-attrdef-decls --attrdefs-dialect=inductor)
mlir_tablegen(InductorAttrs.cpp.inc -gen-attrdef-defs --attrdefs-dialect=inductor)
add_public_tablegen_target(MLIRInductorAttrsIncGen)
add_dependencies(MLIRInductorIncGen MLIRInductorAttrsIncGen)
