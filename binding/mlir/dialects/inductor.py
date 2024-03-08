from mlir.dialects._inductor_ops_gen import *
from mlir.dialects._inductor_enum_gen import *
from mlir._mlir_libs import get_dialect_registry

import mlir.dialects.index


def _import_C_ext():
    import pathlib
    import sysconfig
    import importlib
    import sys
    import ctypes
    ctypes.CDLL("libMLIRPythonCAPI.so", ctypes.RTLD_GLOBAL)
    ctypes.CDLL("libLLVM.so", ctypes.RTLD_GLOBAL)
    ext_path = pathlib.Path(__file__).parent.parent / '_mlir_libs' / (
        '_mlirDialectsInductor' + sysconfig.get_config_var('EXT_SUFFIX'))
    spec = importlib.util.spec_from_file_location(
        'inductor_mlir._mlirDialectsInductor', ext_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules['inductor_mlir._mlirDialectsInductor'] = module
    return module


_C = _import_C_ext()
_C.register_inductor_dialect(get_dialect_registry())

compile_module = _C.compile_module
run_jit = _C.run_jit
