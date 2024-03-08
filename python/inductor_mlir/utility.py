import sympy
import torch

from typing import Optional
from mlir.ir import Context, Module, Location


class BoxedMLIRModule:

    def __init__(self, ctx: Optional[Context] = None):
        if ctx is None:
            self.ctx = Context()
        else:
            self.ctx = ctx
        with self.ctx, Location.unknown():
            self.module = Module.create()


to_plain_offset = sympy.Function('to_plain_offset')


def to_mlir_type(self, dtype):
    if dtype == torch.float32:
        return self.ir_builder.get_f32_type()
    assert False
